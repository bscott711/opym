import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: requires a real CUDA GPU + `module load matlab/R2024b` "
        "(matlab.engine). Not run by default in environments without both.",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden-reference fixtures instead of comparing against them.",
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")
