from dataclasses import dataclass


@dataclass
class PCDSamplerConfig:
    """
    Config class for the sampler.
    """

    dim: int
    number_unit_vectors: int
    number_samples: int
    threshold: float = 0.1
    steps: int = 40
    sorting: bool = True
    # This field means that the initial samples are means of the GM.
    # This is useful for sampling MPC, where we have 1 Gaussian Component per sample
    mean_sampling: bool = False
