import joblib
import pytest

from dropout.data import generate_clients
from dropout.train import train


@pytest.fixture(scope="session")
def clients():
    return generate_clients(n=1500, seed=7)


@pytest.fixture(scope="session")
def bundle(clients):
    return train(clients, make_figures=False, verbose=False)


@pytest.fixture(scope="session")
def model_file(bundle, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "dropout_model.joblib"
    joblib.dump(bundle, path)
    return path
