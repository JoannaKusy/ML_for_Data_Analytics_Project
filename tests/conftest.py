import pytest


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    """
    Forces Weights & Biases to be completely disabled,
    preventing fake data from being logged during tests.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
