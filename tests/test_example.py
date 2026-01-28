import pytest
import torch


def test_example():
    print("This is a placeholder test for the example script.")


@pytest.mark.parametrize("mean", [torch.tensor([0.0, 0.0]), torch.tensor([1.0, -1.0])])
@pytest.mark.parametrize(
    "n_samples",
    [10],
)
def test_mean(mean: torch.Tensor, n_samples: int):
    sample_mean = mean
    assert torch.allclose(mean, mean, atol=1e-6)  # Placeholder assertion
