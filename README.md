This is a library for PCD sampling in python.

## Installation
1. If you wish to use the lib in your project, run: uv add git+https://github.com/KIT-ISAS/PCD_sampling_py.git

## Usage

In order to use PCD sampling in your project you must:

1. Create a config data object
2. Create an instance of sampling mpc class with config injected.
3. Create a Gaussian Mixture to sample from.
4. Use sample method on the class to get the samples. 