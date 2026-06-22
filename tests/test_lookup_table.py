import pytest
import torch
from pcd_sampling_py.lookup_table import LookupTable
from torch.distributions import Normal


@pytest.fixture
def dist():
    components = Normal(torch.tensor([-1.0, 0.5]), torch.tensor([0.7, 1.0]))
    mixture = torch.distributions.Categorical(probs=torch.tensor([0.5, 0.5]))
    return torch.distributions.MixtureSameFamily(mixture, components)


@pytest.fixture
def lut(dist):
    return LookupTable(dist, num_points=1001)


def test_linear_interpolation(lut):
    i = 100

    x0 = lut.xmin + i / lut.inv_dx
    x1 = lut.xmin + (i + 1) / lut.inv_dx

    x = (x0 + x1) / 2

    expected = 0.5 * (lut.table[0, i] + lut.table[0, i + 1])

    actual = lut.pdf_cdf(x)

    assert torch.allclose(actual, expected)

def test_clamps_below_xmin(lut):
    x = lut.xmin - 100.0

    actual = lut.pdf_cdf(x)
    expected = lut.table[0, 0]

    assert torch.allclose(actual, expected)

def test_clamps_above_xmax(lut):
    x = lut.xmax + 100.0

    actual = lut.pdf_cdf(x)
    expected = lut.table[0, -1]

    assert torch.allclose(actual, expected)

def test_output_shape(lut):
    x = torch.randn(4, 5)

    y = lut.pdf_cdf(x)

    assert y.shape == (4, 5, 2)

def test_accuracy(dist):
    lut = LookupTable(dist, num_points=10001)
    x = torch.linspace(lut.xmin[0], lut.xmax[0], 1000)

    approx = lut.pdf_cdf(x)[0, :, :]

    exact = torch.stack(
        (
            torch.exp(dist.log_prob(x)),
            dist.cdf(x),
        ),
        dim=-1,
    )

    assert torch.allclose(
        approx,
        exact,
        atol=1e-5,
        rtol=1e-4,
    )
