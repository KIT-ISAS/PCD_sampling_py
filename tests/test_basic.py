from pcd_sampling_py.sampling_utils import sot_sphere
import torch
from pcd_sampling_py.sampler import PCDSampler
from pcd_sampling_py.models import PCDSamplerConfig


def test_unit_vectors():
    torch.manual_seed(0)

    vectors = sot_sphere(
        20,
        d=2,
        K=200,
        iterations=300,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert vectors.shape == (20, 2)

    norms = torch.norm(vectors, dim=1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    assert torch.sum(vectors, dim=0).mean() < 1e-2  # Check if the sum is close to zero


def test_basic_sampling():
    """
    Test 1D sampling for 1 Gaussian with 30 points.
    """
    torch.manual_seed(0)
    weights = torch.tensor([1.0])
    means = torch.tensor([[0.0]])
    covariances = torch.tensor([[[1.0]]])

    config = PCDSamplerConfig(
        dim=1,
        number_unit_vectors=2000,
        number_samples=30,
        steps=1000,
        sorting=True,
        initial_sampling_method="random",
        unit_vectors_method="random",
    )

    sampler = PCDSampler(config)
    samples = sampler.sample(weights, means, covariances)

    # Moment matching
    assert samples.shape == (30, 1)
    assert torch.allclose(samples.mean(dim=0), means.squeeze(), atol=1e-2)
    assert torch.allclose(
        samples.var(dim=0, unbiased=True), covariances.squeeze(), atol=1e-2
    )
