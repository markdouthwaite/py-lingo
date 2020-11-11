"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import pytest
from tempfile import TemporaryFile
from sklearn.datasets import load_boston, load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    Ridge,
    LinearRegression,
    Lasso,
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from numpy import vstack

from pylingo import load, dump


@pytest.fixture
def boston_dataset():
    dataset = load_boston()
    x = dataset["data"]
    y = dataset["target"]

    x, y = shuffle(x, y)

    return train_test_split(x, y, test_size=0.2)


@pytest.fixture
def iris_dataset():
    dataset = load_iris()
    x = dataset["data"]
    y = dataset["target"]

    x, y = shuffle(x, y)

    return train_test_split(x, y, test_size=0.2)


@pytest.mark.parametrize(
    "model", [Ridge(), LinearRegression(), Lasso(), SGDRegressor()]
)
def test_simple_regressor_dump_load_cycle(model, boston_dataset):
    tx, vx, ty, vy = boston_dataset

    model.fit(tx, ty)

    file = TemporaryFile(suffix=str(model))
    dump(model, file)
    loaded_model = load(file)

    file.close()

    # check that model params are identical
    assert (loaded_model.coef_ - model.coef_).sum() == 0.0
    assert (loaded_model.intercept_ - model.intercept_).sum() == 0.0


@pytest.mark.parametrize(
    "model", [Ridge(), LinearRegression(), Lasso()]
)
def test_multivariate_regressor_dump_load_cycle(model, boston_dataset):
    tx, vx, ty, vy = boston_dataset

    ty = vstack([ty, ty[::-1]]).T

    model.fit(tx, ty)

    file = TemporaryFile(suffix=str(model))
    dump(model, file)
    loaded_model = load(file)

    file.close()

    # check that model params are identical
    assert (loaded_model.coef_ - model.coef_).sum() == 0.0
    assert (loaded_model.intercept_ - model.intercept_).sum() == 0.0


@pytest.mark.parametrize(
    "model", [RidgeClassifier(), LogisticRegression(), SGDClassifier()]
)
def test_simple_classifier_dump_load_cycle(model, iris_dataset):
    tx, vx, ty, vy = iris_dataset

    model.fit(tx, ty)

    file = TemporaryFile(suffix=str(model))
    dump(model, file)
    loaded_model = load(file)

    file.close()

    # check that model params are identical
    assert (loaded_model.coef_ - model.coef_).sum() == 0.0
    assert (loaded_model.intercept_ - model.intercept_).sum() == 0.0
