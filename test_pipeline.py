import pytest
from pipeline import splitData
def test_splitData():
    X_train,y_train,X_test,y_test=splitData()
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]