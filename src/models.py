from dataclasses import dataclass

@dataclass
class PCDSamplerConfig:
    dim: int
    number_unit_vectors: int
    number_samples: int
    threshold: float = 0.1
    steps: int = 40
    sorting: bool = True
