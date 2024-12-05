import numpy as np
from mysklearn.myclassifiers import MyRandomForestClassifier
import pytest

@pytest.fixture
def data():
    """Fixture for setting up training and test data."""
    X_train = [
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1]
    ]
    y_train = ['A', 'A', 'B', 'B', 'A']

    X_test = [
        [1, 0, 0],
        [0, 1, 1]
    ]
    y_test = ['A', 'B']

    return X_train, y_train, X_test, y_test


def test_initialization():
    """Test initialization of the random forest classifier."""
    rf = MyRandomForestClassifier(n_estimators=5, max_features=2, bootstrap=True)
    assert rf.n_estimators == 5
    assert rf.max_features == 2
    assert rf.bootstrap is True

def test_fit(data):
    """Test the fit method to ensure trees are built."""
    X_train, y_train, _, _ = data
    rf = MyRandomForestClassifier(n_estimators=3, max_features=2, bootstrap=True)
    rf.fit(X_train, y_train)
    assert len(rf.trees) == 3  # Check the number of trees

def test_predict(data):
    """Test the predict method."""
    X_train, y_train, X_test, _ = data
    rf = MyRandomForestClassifier(n_estimators=3, max_features=2, bootstrap=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Check the length of predictions
    assert len(y_pred) == len(X_test)

    # Check if predictions are among valid classes
    valid_classes = set(y_train)
    for prediction in y_pred:
        assert prediction in valid_classes
