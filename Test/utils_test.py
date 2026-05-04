# -*- coding: utf-8 -*-
"""
Tests for the utils module.
"""
import pytest
import math
import numpy as np


def test_list_check_equal():
    """Test list_check_equal function."""
    from pycm.utils import list_check_equal

    assert list_check_equal([1, 1, 1]) == True
    assert list_check_equal([1, 2, 1]) == False
    assert list_check_equal(["a", "a"]) == True
    assert list_check_equal([1]) == True  # single element is equal to itself


def test_isfloat():
    """Test isfloat function."""
    from pycm.utils import isfloat

    assert isfloat("3.14") == True
    assert isfloat("123") == True
    assert isfloat("-0.5") == True
    assert isfloat("abc") == False
    assert isfloat("") == False
    assert isfloat(None) == False


def test_rounder():
    """Test rounder function."""
    from pycm.utils import rounder

    assert rounder(3.14159, 2) == "3.14"
    assert rounder(3.14159, 4) == "3.1416"
    assert rounder("None", 5) == "None"
    assert rounder((1, 2, "None"), 5) == "(1,2,None)"
    assert rounder((1.5, 2.5), 1) == "(1.5,2.5)"


def test_class_filter():
    """Test class_filter function."""
    from pycm.utils import class_filter

    classes = [0, 1, 2]
    assert class_filter(classes, [0, 1]) == [0, 1]
    assert class_filter(classes, [0, 3]) == [0, 1, 2]  # 3 not in classes
    assert class_filter(classes, None) == [0, 1, 2]


def test_vector_check():
    """Test vector_check function."""
    from pycm.utils import vector_check

    assert vector_check([1, 2, 3]) == True
    assert vector_check([0, 0, 0]) == True
    assert vector_check([1, 2, 3, 0.4]) == False  # float
    assert vector_check([1, 2, 3, -2]) == False  # negative
    assert vector_check([]) == True  # empty is valid


def test_matrix_check():
    """Test matrix_check function."""
    from pycm.utils import matrix_check

    # Valid matrix
    valid_matrix = {1: {1: 0, 2: 0}, 2: {1: 0, 2: 0}}
    assert matrix_check(valid_matrix) == True

    # Invalid - contains float
    invalid_matrix = {1: {1: 0.5, 2: 0}, 2: {1: 0, 2: 0}}
    assert matrix_check(invalid_matrix) == False

    # Invalid - empty
    assert matrix_check({}) == False
    assert matrix_check([]) == False


def test_vector_filter():
    """Test vector_filter function."""
    from pycm.utils import vector_filter

    # Same types
    actual, predict = vector_filter([1, 2, 3], [1, 2, 3])
    assert actual == [1, 2, 3]
    assert predict == [1, 2, 3]

    # Mixed types should convert to str
    actual, predict = vector_filter([1, 2], ["a", "b"])
    assert actual == ["1", "2"]
    assert predict == ["a", "b"]

    # Numpy arrays
    actual, predict = vector_filter(np.array([1, 2]), np.array([1, 2]))
    assert actual == [1, 2]


def test_class_check():
    """Test class_check function."""
    from pycm.utils import class_check

    assert class_check([1, 2, 3]) == True
    assert class_check(["a", "b", "c"]) == True
    assert class_check([1, "a", 3]) == False  # mixed types


def test_one_vs_all_func():
    """Test one_vs_all_func function."""
    from pycm.utils import one_vs_all_func

    classes = [1, 2]
    table = {1: {1: 0, 2: 0}, 2: {1: 0, 2: 0}}
    TP = {1: 0, 2: 0}
    TN = {1: 0, 2: 0}
    FP = {1: 0, 2: 0}
    FN = {1: 0, 2: 0}

    result_classes, result_table = one_vs_all_func(classes, table, TP, TN, FP, FN, 3)
    # class_name=3 not in TP, so returns original
    assert result_classes == [1, 2]
    assert result_table == table


def test_normalized_table_calc():
    """Test normalized_table_calc function."""
    from pycm.utils import normalized_table_calc

    classes = [0, 1]
    table = {0: {0: 5, 1: 5}, 1: {0: 2, 1: 8}}

    normalized = normalized_table_calc(classes, table)
    assert normalized[0][0] == 0.5
    assert normalized[0][1] == 0.5
    assert normalized[1][0] == 0.2
    assert normalized[1][1] == 0.8


def test_custom_rounder():
    """Test custom_rounder function."""
    from pycm.utils import custom_rounder

    assert custom_rounder(0.1234, 100) == 0.12
    assert custom_rounder(0.1256, 100) == 0.13


def test_transpose_func():
    """Test transpose_func function."""
    from pycm.utils import transpose_func

    classes = [0, 1]
    table = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}

    transposed = transpose_func(classes, table)
    assert transposed[0][1] == 3
    assert transposed[1][0] == 2


def test_matrix_params_from_table():
    """Test matrix_params_from_table function."""
    from pycm.utils import matrix_params_from_table
    from pycm.errors import pycmMatrixError

    table = {0: {0: 5, 1: 2}, 1: {0: 1, 1: 7}}

    classes, new_table, TP, TN, FP, FN = matrix_params_from_table(table)
    assert TP[0] == 5
    assert TP[1] == 7
    assert FN[0] == 2
    assert FN[1] == 1
    assert FP[0] == 1
    assert FP[1] == 2

    # Test with single class
    single_class_table = {0: {0: 5}}
    with pytest.raises(pycmMatrixError):
        matrix_params_from_table(single_class_table)


def test_matrix_params_calc():
    """Test matrix_params_calc function."""
    from pycm.utils import matrix_params_calc

    actual = [0, 0, 1, 1, 1]
    predict = [0, 1, 0, 1, 1]

    classes, table, TP, TN, FP, FN = matrix_params_calc(actual, predict)
    assert 0 in classes
    assert 1 in classes
    assert TP[0] == 1
    assert TP[1] == 2


def test_imbalance_check():
    """Test imbalance_check function."""
    from pycm.utils import imbalance_check

    # Balanced
    balanced_P = {0: 10, 1: 10, 2: 10}
    assert imbalance_check(balanced_P) == False

    # Imbalanced (ratio > 3)
    imbalanced_P = {0: 100, 1: 10}
    assert imbalance_check(imbalanced_P) == True

    # Edge case with 0
    zero_P = {0: 0, 1: 10}
    assert imbalance_check(zero_P) == True


def test_binary_check():
    """Test binary_check function."""
    from pycm.utils import binary_check

    assert binary_check([0, 1]) == True
    assert binary_check([0, 1, 2]) == False
    assert binary_check([0]) == False


def test_complement():
    """Test complement function."""
    from pycm.utils import complement

    assert complement(0.5) == 0.5
    assert complement(0.3) == 0.7
    assert complement("None") == "None"


def test_statistic_recommend():
    """Test statistic_recommend function."""
    from pycm.utils import statistic_recommend
    from pycm.params import BINARY_RECOMMEND, MULTICLASS_RECOMMEND, IMBALANCED_RECOMMEND

    # Binary balanced
    assert statistic_recommend([0, 1], False) == BINARY_RECOMMEND

    # Multiclass balanced
    assert statistic_recommend([0, 1, 2], False) == MULTICLASS_RECOMMEND

    # Imbalanced
    assert statistic_recommend([0, 1], True) == IMBALANCED_RECOMMEND


def test_matrix_combine():
    """Test matrix_combine function."""
    from pycm.utils import matrix_combine

    matrix_1 = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    matrix_2 = {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}}

    combined = matrix_combine(matrix_1, matrix_2)
    assert combined[0][0] == 2
    assert combined[0][1] == 3
    assert combined[1][0] == 4
    assert combined[1][1] == 5


def test_inv_erf():
    """Test inv_erf function."""
    from pycm.utils import inv_erf

    assert inv_erf(-1) == "None"
    assert inv_erf(1) == "None"
    assert inv_erf(-2) == "None"
    assert inv_erf(2) == "None"
    assert inv_erf(0) == 0
    # Test some valid values
    result = inv_erf(0.3)
    assert isinstance(result, float)


def test_sort_char_num():
    """Test sort_char_num function."""
    from pycm.utils import sort_char_num

    input_list = ["SOA10", "SOA1", "SOA2"]
    result = sort_char_num(input_list)
    assert result == ["SOA1", "SOA2", "SOA10"]

    input_list2 = ["A", "B", "A1", "B2"]
    result2 = sort_char_num(input_list2)
    assert result2 == ["A", "A1", "B", "B2"]


def test_vector_serializer():
    """Test vector_serializer function."""
    from pycm.utils import vector_serializer

    # List input
    assert vector_serializer([1, 2, 3]) == [1, 2, 3]

    # Numpy array input
    np_array = np.array([1, 2, 3])
    assert vector_serializer(np_array) == [1, 2, 3]


def test_thresholds_calc():
    """Test thresholds_calc function."""
    from pycm.utils import thresholds_calc

    probs = [0.1, 0.3, 0.5, 0.3, 0.9]
    thresholds = thresholds_calc(probs)
    assert thresholds == [0.1, 0.3, 0.5, 0.9]


def test_threshold_func():
    """Test threshold_func function."""
    from pycm.utils import threshold_func

    classes = ["A", "B"]
    item = [0.7, 0.3]

    # Above threshold
    assert threshold_func(item, 0, classes, 0.5) == "A"

    # Below threshold
    assert threshold_func(item, 0, classes, 0.8) == "B"


def test_deprecated_decorator():
    """Test deprecated decorator."""
    from pycm.utils import deprecated
    import warnings

    @deprecated
    def old_function():
        return "result"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_function()
        assert result == "result"
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "old_function" in str(w[-1].message)


def test_metrics_off_check_decorator():
    """Test metrics_off_check decorator."""
    from pycm.utils import metrics_off_check
    from pycm.errors import pycmMatrixError

    class DummyClass:
        def __init__(self, metrics_off):
            self.metrics_off = metrics_off

        @metrics_off_check
        def some_method(self):
            return "result"

    # When metrics_off is False
    obj_on = DummyClass(False)
    assert obj_on.some_method() == "result"

    # When metrics_off is True
    obj_off = DummyClass(True)
    with pytest.raises(pycmMatrixError):
        obj_off.some_method()


def test_polevl():
    """Test polevl function (polynomial evaluation)."""
    from pycm.utils import polevl

    # Test simple polynomial: 2x^2 + 3x + 1 at x=2
    # 2*(2^2) + 3*2 + 1 = 8 + 6 + 1 = 15
    coefs = [2, 3, 1]
    result = polevl(2, coefs, 2)
    assert result == 15


def test_p1evl():
    """Test p1evl function."""
    from pycm.utils import p1evl

    # Test with coefficient of x^n = 1
    # x^2 + 3x + 1 at x=2
    # 4 + 6 + 1 = 11
    coefs = [3, 1]
    result = p1evl(2, coefs, 2)
    assert result == 11


def test_normal_quantile():
    """Test normal_quantile function."""
    from pycm.utils import normal_quantile

    # Test standard normal distribution
    result = normal_quantile(0.5)
    assert abs(result) < 0.001  # Should be close to 0


def test_sparse_matrix_calc():
    """Test sparse_matrix_calc function."""
    from pycm.utils import sparse_matrix_calc

    classes = [0, 1, 2]
    # Class 2 has all zeros in row and column
    table = {
        0: {0: 5, 1: 2, 2: 0},
        1: {0: 1, 1: 7, 2: 0},
        2: {0: 0, 1: 0, 2: 0}
    }

    sparse_table, actual_classes, predict_classes = sparse_matrix_calc(classes, table)
    assert 2 not in actual_classes  # Row with all zeros removed
    assert 2 not in predict_classes  # Column with all zeros removed
