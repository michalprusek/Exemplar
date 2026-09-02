"""Shared pytest fixtures. Local suite runs CPU-only with no model downloads;
GPU/model tests are marked ``@pytest.mark.gpu`` and excluded by default (see
pyproject ``addopts``)."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
