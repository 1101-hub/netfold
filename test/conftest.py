"""
Shared fixtures for NetFold tests.
"""

import os
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def test_dir():
    return TEST_DIR


@pytest.fixture
def cube_obj(test_dir):
    return os.path.join(test_dir, "cube.obj")


@pytest.fixture
def icosahedron_obj(test_dir):
    return os.path.join(test_dir, "icosahedron.obj")


@pytest.fixture
def geodesic_obj(test_dir):
    """Returns a function that gives the path for geodesic_nN.obj."""
    def _path(n: int) -> str:
        return os.path.join(test_dir, f"geodesic_n{n}.obj")
    return _path
