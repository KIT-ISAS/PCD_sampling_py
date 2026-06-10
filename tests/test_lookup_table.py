import pytest
import torch
from pcd_sampling_py.lookup_table import LookupTable
from torch.distributions import Normal


@pytest.fixture
def dist():
    return Normal(0.0, 1.0)


@pytest.fixture
def lut(dist):
    return LookupTable(dist, num_points=1001)


def test_exact_grid_points(lut):
    grid = torch.linspace(
        lut.xmin,
        lut.xmax,
        lut.num_points,
        device=lut.table.device,
        dtype=lut.table.dtype,
    )

    values = lut.pdf_cdf(grid)

    assert torch.allclose(values, lut.table)


def test_linear_interpolation(lut):
    i = 100

    x0 = lut.xmin + i / lut.inv_dx
    x1 = lut.xmin + (i + 1) / lut.inv_dx

    x = torch.tensor((x0 + x1) / 2)

    expected = 0.5 * (lut.table[i] + lut.table[i + 1])

    actual = lut.pdf_cdf(x)

    assert torch.allclose(actual, expected)

def test_clamps_below_xmin(lut):
    x = torch.tensor(lut.xmin - 100.0)

    actual = lut.pdf_cdf(x)
    expected = lut.table[0]

    assert torch.allclose(actual, expected)

def test_clamps_above_xmax(lut):
    x = torch.tensor(lut.xmax + 100.0)

    actual = lut.pdf_cdf(x)
    expected = lut.table[-1]

    assert torch.allclose(actual, expected)

def test_output_shape(lut):
    x = torch.randn(4, 5)

    y = lut.pdf_cdf(x)

    assert y.shape == (4, 5, 2)

def test_accuracy(dist):
    lut = LookupTable(
        dist,
        num_points=10001,
        xmin=-5,
        xmax=5,
    )

    x = torch.linspace(-4.9, 4.9, 1000)

    approx = lut.pdf_cdf(x)

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
