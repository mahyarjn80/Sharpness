# Sharpness Optimizer Implementation

This repository includes an implementation of various optimizers to study sharpness properties of neural networks.

## Muon Optimizer

The Muon optimizer has been integrated into the framework. Muon (MomentUm Orthogonalized by Newton-schulz) is an optimizer 
designed specifically for hidden layers of neural networks. It performs an orthogonalization post-processing step on 
the standard SGD-momentum updates using a Newton-Schulz iteration.

### Using the Muon Optimizer

You can use the Muon optimizer by passing `--opt muon` to the training scripts:

```bash
python src/gd.py [dataset] [architecture] [loss] [learning_rate] [max_steps] --opt muon --beta 0.95
```

### Implementation Details

- The Muon implementation is adapted from the original implementation in `src/muon/muon.py`.
- It has been simplified to work in single-GPU mode by default.
- The optimizer works by grouping parameters by size and applying the Newton-Schulz iteration to orthogonalize updates.

### Recommendations

- Use a higher learning rate than you would with standard SGD (e.g., 0.02 instead of 0.001).
- A momentum value of 0.95 is recommended, with Nesterov acceleration enabled by default.
- The Muon optimizer is most effective for hidden layers and may not be suitable for embedding or output layers.
- For convolutional layers, the 4D filters are flattened to 2D before orthogonalization.

### Original Implementation

The original Muon optimizer implementation can be found at [kellerjordan/muon](https://github.com/kellerjordan/muon).
More information about the Muon optimizer can be found in the blog post: [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/). 