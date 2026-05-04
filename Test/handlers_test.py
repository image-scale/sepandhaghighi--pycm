# -*- coding: utf-8 -*-
"""Tests for the handlers module."""
import pytest
import json
import tempfile
import os
import numpy as np


class MockConfusionMatrix:
    """Mock ConfusionMatrix class for testing handlers."""

    def __init__(self):
        self.classes = None
        self.table = None
        self.matrix = None
        self.normalized_table = None
        self.normalized_matrix = None
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.P = None
        self.class_stat = {}
        self.overall_stat = {}
        self.metrics_off = False
        self.imbalance = None
        self.transpose = False
        self.digit = 5
        self.actual_vector = None
        self.predict_vector = None
        self.prob_vector = None
        self.weights = None


def test_class_stat_init():
    """Test __class_stat_init__ function."""
    from pycm.handlers import __class_stat_init__
    from pycm.params import CLASS_PARAMS

    cm = MockConfusionMatrix()
    cm.class_stat = {stat: {0: 1, 1: 2} for stat in CLASS_PARAMS}

    __class_stat_init__(cm)

    # Check that attributes are set
    for stat, field_name in CLASS_PARAMS.items():
        assert hasattr(cm, field_name)
        assert getattr(cm, field_name) == {0: 1, 1: 2}


def test_overall_stat_init():
    """Test __overall_stat_init__ function."""
    from pycm.handlers import __overall_stat_init__
    from pycm.params import OVERALL_PARAMS

    cm = MockConfusionMatrix()
    cm.overall_stat = {stat: 0.5 for stat in OVERALL_PARAMS}

    __overall_stat_init__(cm)

    # Check that attributes are set
    for stat, field_name in OVERALL_PARAMS.items():
        assert hasattr(cm, field_name)
        assert getattr(cm, field_name) == 0.5


def test_imbalancement_handler_none():
    """Test __imbalancement_handler__ when imbalance is None."""
    from pycm.handlers import __imbalancement_handler__

    cm = MockConfusionMatrix()
    cm.imbalance = None
    cm.P = {0: 10, 1: 10}  # Balanced

    __imbalancement_handler__(cm, None)

    assert cm.imbalance == False  # Balanced case


def test_imbalancement_handler_imbalanced():
    """Test __imbalancement_handler__ with imbalanced data."""
    from pycm.handlers import __imbalancement_handler__

    cm = MockConfusionMatrix()
    cm.imbalance = None
    cm.P = {0: 100, 1: 10}  # Imbalanced

    __imbalancement_handler__(cm, None)

    assert cm.imbalance == True


def test_imbalancement_handler_passed_flag():
    """Test __imbalancement_handler__ with passed flag."""
    from pycm.handlers import __imbalancement_handler__

    cm = MockConfusionMatrix()
    cm.imbalance = None
    cm.P = {0: 10, 1: 10}

    __imbalancement_handler__(cm, True)

    assert cm.imbalance == True  # Uses passed flag


def test_imbalancement_handler_already_set():
    """Test __imbalancement_handler__ when imbalance already set."""
    from pycm.handlers import __imbalancement_handler__

    cm = MockConfusionMatrix()
    cm.imbalance = True  # Already set
    cm.P = {0: 10, 1: 10}

    __imbalancement_handler__(cm, False)

    assert cm.imbalance == True  # Unchanged


def test_obj_assign_handler_basic():
    """Test __obj_assign_handler__ with basic parameters."""
    from pycm.handlers import __obj_assign_handler__

    cm = MockConfusionMatrix()
    cm.metrics_off = True  # Disable metrics for simpler test

    classes = [0, 1]
    table = {0: {0: 5, 1: 2}, 1: {0: 1, 1: 7}}
    TP = {0: 5, 1: 7}
    TN = {0: 7, 1: 5}
    FP = {0: 1, 1: 2}
    FN = {0: 2, 1: 1}

    matrix_param = (classes, table, TP, TN, FP, FN)

    __obj_assign_handler__(cm, matrix_param)

    assert cm.classes == [0, 1]
    assert cm.table == table
    assert cm.matrix == table
    assert cm.TP == TP
    assert cm.TN == TN
    assert cm.FP == FP
    assert cm.FN == FN
    assert cm.normalized_table is not None


def test_obj_assign_handler_with_metrics():
    """Test __obj_assign_handler__ with metrics calculation."""
    from pycm.handlers import __obj_assign_handler__

    cm = MockConfusionMatrix()
    cm.metrics_off = False

    classes = [0, 1]
    table = {0: {0: 5, 1: 2}, 1: {0: 1, 1: 7}}
    TP = {0: 5, 1: 7}
    TN = {0: 7, 1: 5}
    FP = {0: 1, 1: 2}
    FN = {0: 2, 1: 1}

    matrix_param = (classes, table, TP, TN, FP, FN)

    __obj_assign_handler__(cm, matrix_param)

    assert cm.classes == [0, 1]
    assert "TPR" in cm.class_stat
    assert "Overall ACC" in cm.overall_stat


def test_obj_matrix_handler_valid():
    """Test __obj_matrix_handler__ with valid matrix."""
    from pycm.handlers import __obj_matrix_handler__

    matrix = {0: {0: 5, 1: 2}, 1: {0: 1, 1: 7}}

    result = __obj_matrix_handler__(matrix, None, False)

    assert len(result) == 6
    assert 0 in result[0]
    assert 1 in result[0]


def test_obj_matrix_handler_with_classes():
    """Test __obj_matrix_handler__ with explicit classes."""
    from pycm.handlers import __obj_matrix_handler__

    matrix = {"A": {"A": 5, "B": 2}, "B": {"A": 1, "B": 7}}

    result = __obj_matrix_handler__(matrix, ["A", "B"], False)

    assert result[0] == ["A", "B"]


def test_obj_matrix_handler_invalid_matrix():
    """Test __obj_matrix_handler__ with invalid matrix."""
    from pycm.handlers import __obj_matrix_handler__
    from pycm.errors import pycmMatrixError

    # Empty matrix
    with pytest.raises(pycmMatrixError):
        __obj_matrix_handler__({}, None, False)

    # Invalid values
    invalid_matrix = {0: {0: 1.5, 1: 2}, 1: {0: 1, 1: 2}}
    with pytest.raises(pycmMatrixError):
        __obj_matrix_handler__(invalid_matrix, None, False)


def test_obj_matrix_handler_mixed_class_types():
    """Test __obj_matrix_handler__ with mixed class types."""
    from pycm.handlers import __obj_matrix_handler__
    from pycm.errors import pycmMatrixError

    # Mixed types in keys
    mixed_matrix = {0: {0: 1, "a": 2}, "a": {0: 1, "a": 2}}
    with pytest.raises(pycmMatrixError):
        __obj_matrix_handler__(mixed_matrix, None, False)


def test_obj_array_handler_valid():
    """Test __obj_array_handler__ with valid array."""
    from pycm.handlers import __obj_array_handler__

    array = [[5, 2], [1, 7]]

    result = __obj_array_handler__(array, None, False)

    assert len(result) == 6
    assert result[0] == [0, 1]


def test_obj_array_handler_with_classes():
    """Test __obj_array_handler__ with explicit classes."""
    from pycm.handlers import __obj_array_handler__

    array = [[5, 2], [1, 7]]

    result = __obj_array_handler__(array, ["A", "B"], False)

    assert result[0] == ["A", "B"]


def test_obj_array_handler_numpy():
    """Test __obj_array_handler__ with numpy array."""
    from pycm.handlers import __obj_array_handler__

    array = np.array([[5, 2], [1, 7]])

    result = __obj_array_handler__(array, None, False)

    assert len(result) == 6


def test_obj_array_handler_invalid_classes_length():
    """Test __obj_array_handler__ with wrong classes length."""
    from pycm.handlers import __obj_array_handler__
    from pycm.errors import pycmMatrixError

    array = [[5, 2], [1, 7]]

    with pytest.raises(pycmMatrixError):
        __obj_array_handler__(array, ["A", "B", "C"], False)


def test_obj_array_handler_duplicate_classes():
    """Test __obj_array_handler__ with duplicate classes."""
    from pycm.handlers import __obj_array_handler__
    from pycm.errors import pycmMatrixError

    array = [[5, 2], [1, 7]]

    with pytest.raises(pycmMatrixError):
        __obj_array_handler__(array, ["A", "A"], False)


def test_obj_vector_handler_valid():
    """Test __obj_vector_handler__ with valid vectors."""
    from pycm.handlers import __obj_vector_handler__

    cm = MockConfusionMatrix()
    actual = [0, 0, 1, 1]
    predict = [0, 1, 0, 1]

    result = __obj_vector_handler__(cm, actual, predict, None, None, None)

    assert len(result) == 6


def test_obj_vector_handler_numpy():
    """Test __obj_vector_handler__ with numpy arrays."""
    from pycm.handlers import __obj_vector_handler__

    cm = MockConfusionMatrix()
    actual = np.array([0, 0, 1, 1])
    predict = np.array([0, 1, 0, 1])

    result = __obj_vector_handler__(cm, actual, predict, None, None, None)

    assert len(result) == 6


def test_obj_vector_handler_with_weights():
    """Test __obj_vector_handler__ with sample weights."""
    from pycm.handlers import __obj_vector_handler__

    cm = MockConfusionMatrix()
    actual = [0, 0, 1, 1]
    predict = [0, 1, 0, 1]
    weights = [1, 1, 2, 2]

    result = __obj_vector_handler__(cm, actual, predict, None, weights, None)

    assert cm.weights == weights


def test_obj_vector_handler_with_threshold():
    """Test __obj_vector_handler__ with threshold function."""
    from pycm.handlers import __obj_vector_handler__

    cm = MockConfusionMatrix()
    actual = [0, 0, 1, 1]
    # Probabilities
    predict = [[0.7, 0.3], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]]

    def threshold_func(probs):
        return 1 if probs[1] > 0.5 else 0

    result = __obj_vector_handler__(cm, actual, predict, threshold_func, None, None)

    assert cm.prob_vector == predict
    assert cm.predict_vector == [0, 1, 1, 1]


def test_obj_vector_handler_invalid_type():
    """Test __obj_vector_handler__ with invalid type."""
    from pycm.handlers import __obj_vector_handler__
    from pycm.errors import pycmVectorError

    cm = MockConfusionMatrix()

    with pytest.raises(pycmVectorError):
        __obj_vector_handler__(cm, "invalid", [0, 1], None, None, None)


def test_obj_vector_handler_size_mismatch():
    """Test __obj_vector_handler__ with size mismatch."""
    from pycm.handlers import __obj_vector_handler__
    from pycm.errors import pycmVectorError

    cm = MockConfusionMatrix()

    with pytest.raises(pycmVectorError):
        __obj_vector_handler__(cm, [0, 0, 1], [0, 1], None, None, None)


def test_obj_vector_handler_empty():
    """Test __obj_vector_handler__ with empty vectors."""
    from pycm.handlers import __obj_vector_handler__
    from pycm.errors import pycmVectorError

    cm = MockConfusionMatrix()

    with pytest.raises(pycmVectorError):
        __obj_vector_handler__(cm, [], [], None, None, None)


def test_obj_vector_handler_duplicate_classes():
    """Test __obj_vector_handler__ with duplicate classes."""
    from pycm.handlers import __obj_vector_handler__
    from pycm.errors import pycmVectorError

    cm = MockConfusionMatrix()

    with pytest.raises(pycmVectorError):
        __obj_vector_handler__(cm, [0, 1], [0, 1], None, None, ["A", "A"])


def test_obj_file_handler_with_vectors():
    """Test __obj_file_handler__ with vector data."""
    from pycm.handlers import __obj_file_handler__

    cm = MockConfusionMatrix()

    file_data = {
        "Actual-Vector": [0, 0, 1, 1],
        "Predict-Vector": [0, 1, 0, 1],
        "Digit": 5
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(file_data, f)
        temp_path = f.name

    try:
        with open(temp_path, 'r') as f:
            result = __obj_file_handler__(cm, f)

        assert len(result) == 6
        assert cm.actual_vector == [0, 0, 1, 1]
        assert cm.predict_vector == [0, 1, 0, 1]
        assert cm.digit == 5
    finally:
        os.unlink(temp_path)


def test_obj_file_handler_with_matrix():
    """Test __obj_file_handler__ with matrix data."""
    from pycm.handlers import __obj_file_handler__

    cm = MockConfusionMatrix()

    file_data = {
        "Actual-Vector": None,
        "Predict-Vector": None,
        "Matrix": {"0": {"0": 5, "1": 2}, "1": {"0": 1, "1": 7}},
        "Digit": 3
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(file_data, f)
        temp_path = f.name

    try:
        with open(temp_path, 'r') as f:
            result = __obj_file_handler__(cm, f)

        assert len(result) == 6
        assert cm.digit == 3
    finally:
        os.unlink(temp_path)


def test_obj_file_handler_with_weights():
    """Test __obj_file_handler__ with sample weights."""
    from pycm.handlers import __obj_file_handler__

    cm = MockConfusionMatrix()

    file_data = {
        "Actual-Vector": [0, 0, 1, 1],
        "Predict-Vector": [0, 1, 0, 1],
        "Sample-Weight": [1, 1, 2, 2],
        "Digit": 5
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(file_data, f)
        temp_path = f.name

    try:
        with open(temp_path, 'r') as f:
            result = __obj_file_handler__(cm, f)

        assert cm.weights == [1, 1, 2, 2]
    finally:
        os.unlink(temp_path)


def test_all_handlers_importable():
    """Test that all handler functions can be imported."""
    from pycm.handlers import (
        __class_stat_init__,
        __overall_stat_init__,
        __imbalancement_handler__,
        __obj_assign_handler__,
        __obj_file_handler__,
        __obj_matrix_handler__,
        __obj_array_handler__,
        __obj_vector_handler__
    )

    # All functions should be callable
    assert callable(__class_stat_init__)
    assert callable(__overall_stat_init__)
    assert callable(__imbalancement_handler__)
    assert callable(__obj_assign_handler__)
    assert callable(__obj_file_handler__)
    assert callable(__obj_matrix_handler__)
    assert callable(__obj_array_handler__)
    assert callable(__obj_vector_handler__)
