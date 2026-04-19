import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="Run tests marked as @pytest.mark.heavy (expensive, skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--heavy"):
        return
    skip = pytest.mark.skip(reason="needs --heavy option to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip)

