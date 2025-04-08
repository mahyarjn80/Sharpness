import torch
import torch.distributed as dist
from torch import Tensor

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class MuonOptimizer(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        # Adapted to work in single GPU mode by default 
        rank = 0
        world_size = 1
        
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = [p for p in params]
        param_groups = []
        
        # Handle parameters in groups by size
        for size in {p.numel() for p in params}:
            # In single-GPU mode, we just need a buffer for the current GPU
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer = group["update_buffer"]
            update_buffer_views = group["update_buffer_views"]
            # Generate weight updates
            params = group["params"]
            handle = None
            params_world = None
            
            # Since we're operating in single-GPU mode, we simplify the distributed operations
            def update_prev():
                if handle is not None:  # This check is for the first iteration
                    handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    # Calculate scaling factor based on tensor dimensionality
                    scaling = 1.0
                    if p_world.ndim >= 2:
                        scaling = max(1, p_world.size(-2) / p_world.size(-1))**0.5
                    p_world.add_(g_world.view_as(p_world), alpha=-group["lr"] * scaling)
            
            # Process parameters in batches
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        continue
                    
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    
                    # Skip orthogonalization for parameters with fewer than 2 dimensions
                    if g.ndim < 2:
                        # Use the standard SGD update for these parameters
                        update_buffer_views[0].copy_(g.flatten())
                    else:
                        # Apply orthogonalization for 2D+ parameters
                        if g.ndim == 4:  # For conv filters
                            g = g.view(len(g), -1)
                        
                        # Apply Newton-Schulz iteration
                        g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                
                if base_i > 0:
                    update_prev()
                    
                # In single GPU mode, we don't need all_gather, just copy the data
                update_buffer_views[0].copy_(g)
                handle = torch.futures.Future()
                handle.set_result(None)  # Immediately mark as done
                params_world = params[base_i : min(base_i + self.world_size, len(params))]
            
            update_prev() 