import time
import torch

from src.models import PCDSamplingConfig
from src.pcd_sampling import PCDSamplingStrategy
from src.models import GaussianMixture


def sample():
    original_gm = GaussianMixture(
        weights=torch.tensor([0.5, 0.5]),
        means=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        covariances=torch.tensor([[[3.0, 2.8], [2.8, 3.0]], [[3.0, -2.8], [-2.8, 3.0]]]),
    )

    torch.manual_seed(42)
    
    sum_time = 0.0

    sampling_config = PCDSamplingConfig(number_samples=1500, dim=2, number_unit_vectors=100, threshold=0.0001, steps=1000, sorting=True)

    sampler = PCDSamplingStrategy(sampling_config)
    
    for i in range(20):
        start = time.time()
        X = sampler.sample(original_gm)
        diff = time.time() - start
        if i !=0 :
            sum_time += diff
            
        print(f"elapsed: {diff}")
    print(f"Total no comp: {sum_time}")

    


if __name__ == "__main__":
    torch.set_default_device("cuda")
    with torch.no_grad():
        sample()
