import pytest
from samples.calculator.calculator import divide


def test_divide_normal():
    assert divide(10, 2) == 5.0


def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)


def test_divide_negative():
    assert divide(-6, 3) == -2.0
