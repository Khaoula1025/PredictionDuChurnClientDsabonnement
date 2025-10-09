import pytest
from pipeline import prepareDataset
def test_prepareDataset():
    X_train, X_test, y_train, y_test = prepareDataset()
    
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

