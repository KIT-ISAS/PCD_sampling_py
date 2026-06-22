from dataclasses import dataclass
from typing import Literal


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
    lookup_table: bool = True
    local_update: bool = True
   
    # We can take initial samples by the mean or with ut so that they converge faster to desired distribution.
    initial_sampling_method: Literal["mean", "random", "ut"] = "random"
    # Take lcd for better unit sphere coverage (Overhead for the very first sample call)
    unit_vectors_method: Literal["random", "deterministic"] = "random"