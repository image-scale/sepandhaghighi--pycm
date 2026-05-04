# -*- coding: utf-8 -*-
"""
Tests for the errors module.
"""
import pytest


def test_pycm_vector_error():
    """Test pycmVectorError class."""
    from pycm.errors import pycmVectorError

    # Test that it's a subclass of Exception
    assert issubclass(pycmVectorError, Exception)

    # Test raising the error
    with pytest.raises(pycmVectorError):
        raise pycmVectorError("Test vector error")

    # Test error message
    try:
        raise pycmVectorError("Vector size mismatch")
    except pycmVectorError as e:
        assert str(e) == "Vector size mismatch"


def test_pycm_matrix_error():
    """Test pycmMatrixError class."""
    from pycm.errors import pycmMatrixError

    assert issubclass(pycmMatrixError, Exception)

    with pytest.raises(pycmMatrixError):
        raise pycmMatrixError("Test matrix error")

    try:
        raise pycmMatrixError("Invalid matrix format")
    except pycmMatrixError as e:
        assert str(e) == "Invalid matrix format"


def test_pycm_ci_error():
    """Test pycmCIError class."""
    from pycm.errors import pycmCIError

    assert issubclass(pycmCIError, Exception)

    with pytest.raises(pycmCIError):
        raise pycmCIError("Test CI error")

    try:
        raise pycmCIError("Unsupported CI parameter")
    except pycmCIError as e:
        assert str(e) == "Unsupported CI parameter"


def test_pycm_average_error():
    """Test pycmAverageError class."""
    from pycm.errors import pycmAverageError

    assert issubclass(pycmAverageError, Exception)

    with pytest.raises(pycmAverageError):
        raise pycmAverageError("Test average error")

    try:
        raise pycmAverageError("Invalid parameter for average")
    except pycmAverageError as e:
        assert str(e) == "Invalid parameter for average"


def test_pycm_compare_error():
    """Test pycmCompareError class."""
    from pycm.errors import pycmCompareError

    assert issubclass(pycmCompareError, Exception)

    with pytest.raises(pycmCompareError):
        raise pycmCompareError("Test compare error")

    try:
        raise pycmCompareError("At least 2 confusion matrices required")
    except pycmCompareError as e:
        assert str(e) == "At least 2 confusion matrices required"


def test_pycm_plot_error():
    """Test pycmPlotError class."""
    from pycm.errors import pycmPlotError

    assert issubclass(pycmPlotError, Exception)

    with pytest.raises(pycmPlotError):
        raise pycmPlotError("Test plot error")

    try:
        raise pycmPlotError("Failed to import matplotlib")
    except pycmPlotError as e:
        assert str(e) == "Failed to import matplotlib"


def test_pycm_curve_error():
    """Test pycmCurveError class."""
    from pycm.errors import pycmCurveError

    assert issubclass(pycmCurveError, Exception)

    with pytest.raises(pycmCurveError):
        raise pycmCurveError("Test curve error")

    try:
        raise pycmCurveError("Invalid thresholds")
    except pycmCurveError as e:
        assert str(e) == "Invalid thresholds"


def test_pycm_multilabel_error():
    """Test pycmMultiLabelError class."""
    from pycm.errors import pycmMultiLabelError

    assert issubclass(pycmMultiLabelError, Exception)

    with pytest.raises(pycmMultiLabelError):
        raise pycmMultiLabelError("Test multilabel error")

    try:
        raise pycmMultiLabelError("Invalid multilabel input")
    except pycmMultiLabelError as e:
        assert str(e) == "Invalid multilabel input"


def test_all_error_classes_distinct():
    """Test that all error classes are distinct."""
    from pycm.errors import (
        pycmVectorError, pycmMatrixError, pycmCIError,
        pycmAverageError, pycmCompareError, pycmPlotError,
        pycmCurveError, pycmMultiLabelError
    )

    error_classes = [
        pycmVectorError, pycmMatrixError, pycmCIError,
        pycmAverageError, pycmCompareError, pycmPlotError,
        pycmCurveError, pycmMultiLabelError
    ]

    # All classes should be unique
    assert len(error_classes) == len(set(error_classes))

    # All classes should be subclasses of Exception
    for cls in error_classes:
        assert issubclass(cls, Exception)


def test_error_inheritance():
    """Test that catching Exception catches all custom errors."""
    from pycm.errors import pycmVectorError, pycmMatrixError

    # Should be able to catch with Exception
    try:
        raise pycmVectorError("test")
    except Exception as e:
        assert isinstance(e, pycmVectorError)

    try:
        raise pycmMatrixError("test")
    except Exception as e:
        assert isinstance(e, pycmMatrixError)
