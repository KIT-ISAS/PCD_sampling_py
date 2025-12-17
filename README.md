A python Library for Deterministic sampling based on Projected Cumulative Distribution

Original paper: https://ieeexplore.ieee.org/abstract/document/9086228

## Installation
If you wish to use the lib in your project, run: uv add git+https://github.com/KIT-ISAS/PCD_sampling_py.git

## Usage

In order to use PCD sampling in your project you must:

1. Create an iunstance of config data object
2. Create an instance of sampling PCD class with config injected.
3. Create a Gaussian Mixture to sample from.
4. Use sample method on the class to get the samples from the GM.

## Example 
``` python
    torch.manual_seed(42) # for reproducibility

    gm = GaussianMixture(
        weights=torch.tensor([0.5, 0.5]),
        means=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        covariances=torch.tensor([[[3.0, 2.8], [2.8, 3.0]], [[3.0, -2.8], [-2.8, 3.0]]]),
    ) # Create a Gaussian mixture

    
    sampling_config = PCDSamplingConfig(number_samples=40, dim=2, number_unit_vectors=100, threshold=0.0001, steps=100, sorting=True) # Create a config

    sampler = PCDSamplingStrategy(sampling_config) # Inject config into the sampling class

    samples = sampler.sample(gm) # Sample
```

## Important
When you first call sample pytorch compiles it, which takes relatively long time. So it is advised to warm up first by sampling a dummy GM.