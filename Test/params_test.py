# -*- coding: utf-8 -*-
"""
Tests for the params module.
"""
import pytest


def test_pycm_version():
    """Test that PYCM_VERSION is defined and is a string."""
    from pycm.params import PYCM_VERSION
    assert isinstance(PYCM_VERSION, str)
    assert PYCM_VERSION == "4.6"


def test_overview():
    """Test that OVERVIEW contains expected content."""
    from pycm.params import OVERVIEW
    assert "PyCM" in OVERVIEW
    assert "confusion matrix" in OVERVIEW
    assert "https://doi.org/10.21105/joss.00729" in OVERVIEW


def test_error_messages():
    """Test that error messages are defined."""
    from pycm.params import (
        VECTOR_TYPE_ERROR, VECTOR_SIZE_ERROR, VECTOR_EMPTY_ERROR,
        MATRIX_FORMAT_ERROR, COMPARE_TYPE_ERROR, COMPARE_NUMBER_ERROR
    )
    assert isinstance(VECTOR_TYPE_ERROR, str)
    assert isinstance(VECTOR_SIZE_ERROR, str)
    assert isinstance(VECTOR_EMPTY_ERROR, str)
    assert isinstance(MATRIX_FORMAT_ERROR, str)
    assert isinstance(COMPARE_TYPE_ERROR, str)
    assert isinstance(COMPARE_NUMBER_ERROR, str)


def test_class_params():
    """Test CLASS_PARAMS dictionary."""
    from pycm.params import CLASS_PARAMS
    assert isinstance(CLASS_PARAMS, dict)
    assert "TPR" in CLASS_PARAMS
    assert "TNR" in CLASS_PARAMS
    assert "PPV" in CLASS_PARAMS
    assert "F1" in CLASS_PARAMS
    assert "MCC" in CLASS_PARAMS
    assert "AUC" in CLASS_PARAMS


def test_overall_params():
    """Test OVERALL_PARAMS dictionary."""
    from pycm.params import OVERALL_PARAMS
    assert isinstance(OVERALL_PARAMS, dict)
    assert "Overall ACC" in OVERALL_PARAMS
    assert "Kappa" in OVERALL_PARAMS
    assert "Cramer V" in OVERALL_PARAMS


def test_benchmark_scores():
    """Test benchmark score dictionaries."""
    from pycm.params import (
        PLRI_SCORE, NLRI_SCORE, DPI_SCORE, AUCI_SCORE,
        SOA1_SCORE, SOA2_SCORE, SOA3_SCORE, SOA4_SCORE
    )
    # Check PLRI_SCORE
    assert PLRI_SCORE["Good"] == 4
    assert PLRI_SCORE["Fair"] == 3
    assert PLRI_SCORE["Poor"] == 2
    assert PLRI_SCORE["Negligible"] == 1
    assert PLRI_SCORE["None"] == "None"

    # Check AUCI_SCORE
    assert AUCI_SCORE["Excellent"] == 5
    assert AUCI_SCORE["Very Good"] == 4
    assert AUCI_SCORE["Good"] == 3

    # Check SOA1_SCORE (Landis & Koch)
    assert SOA1_SCORE["Almost Perfect"] == 6
    assert SOA1_SCORE["Substantial"] == 5


def test_class_benchmark_list():
    """Test CLASS_BENCHMARK_LIST is sorted."""
    from pycm.params import CLASS_BENCHMARK_LIST, CLASS_BENCHMARK_SCORE_DICT
    assert CLASS_BENCHMARK_LIST == sorted(CLASS_BENCHMARK_SCORE_DICT)
    assert "AUCI" in CLASS_BENCHMARK_LIST
    assert "PLRI" in CLASS_BENCHMARK_LIST


def test_overall_benchmark_list():
    """Test OVERALL_BENCHMARK_LIST is sorted."""
    from pycm.params import OVERALL_BENCHMARK_LIST, OVERALL_BENCHMARK_SCORE_DICT
    assert OVERALL_BENCHMARK_LIST == sorted(OVERALL_BENCHMARK_SCORE_DICT)
    assert "SOA1" in OVERALL_BENCHMARK_LIST
    assert "SOA6" in OVERALL_BENCHMARK_LIST


def test_alpha_tables():
    """Test alpha confidence interval tables."""
    from pycm.params import ALPHA_TWO_SIDE_TABLE, ALPHA_ONE_SIDE_TABLE
    # Two-sided
    assert 0.05 in ALPHA_TWO_SIDE_TABLE
    assert ALPHA_TWO_SIDE_TABLE[0.05] == 1.96

    # One-sided
    assert 0.05 in ALPHA_ONE_SIDE_TABLE
    assert ALPHA_ONE_SIDE_TABLE[0.05] == 1.645


def test_ci_lists():
    """Test CI parameter lists."""
    from pycm.params import CI_CLASS_LIST, CI_OVERALL_LIST
    assert "TPR" in CI_CLASS_LIST
    assert "AUC" in CI_CLASS_LIST
    assert "Kappa" in CI_OVERALL_LIST
    assert "Overall ACC" in CI_OVERALL_LIST


def test_table_colors():
    """Test TABLE_COLOR dictionary."""
    from pycm.params import TABLE_COLOR
    assert isinstance(TABLE_COLOR, dict)
    assert TABLE_COLOR["red"] == [255, 0, 0]
    assert TABLE_COLOR["green"] == [0, 128, 0]
    assert TABLE_COLOR["blue"] == [0, 0, 255]
    assert TABLE_COLOR["white"] == [255, 255, 255]
    assert TABLE_COLOR["black"] == [0, 0, 0]


def test_params_description():
    """Test PARAMS_DESCRIPTION dictionary."""
    from pycm.params import PARAMS_DESCRIPTION
    assert isinstance(PARAMS_DESCRIPTION, dict)
    assert "TPR" in PARAMS_DESCRIPTION
    assert "sensitivity" in PARAMS_DESCRIPTION["TPR"]
    assert "TNR" in PARAMS_DESCRIPTION
    assert "specificity" in PARAMS_DESCRIPTION["TNR"]


def test_params_link():
    """Test PARAMS_LINK dictionary."""
    from pycm.params import PARAMS_LINK
    assert isinstance(PARAMS_LINK, dict)
    assert "TPR" in PARAMS_LINK
    assert "Overall ACC" in PARAMS_LINK


def test_html_templates():
    """Test HTML templates."""
    from pycm.params import HTML_INIT_TEMPLATE, HTML_END_TEMPLATE, HTML_DATASET_TYPE_TEMPLATE
    assert "<!doctype html>" in HTML_INIT_TEMPLATE
    assert "{description}" in HTML_INIT_TEMPLATE
    assert "{version}" in HTML_END_TEMPLATE
    assert "{balance_type}" in HTML_DATASET_TYPE_TEMPLATE


def test_thresholds():
    """Test threshold constants."""
    from pycm.params import CLASS_NUMBER_THRESHOLD, BALANCE_RATIO_THRESHOLD
    assert CLASS_NUMBER_THRESHOLD == 10
    assert BALANCE_RATIO_THRESHOLD == 3


def test_recommend_lists():
    """Test recommendation lists."""
    from pycm.params import BINARY_RECOMMEND, MULTICLASS_RECOMMEND, IMBALANCED_RECOMMEND
    assert isinstance(BINARY_RECOMMEND, list)
    assert isinstance(MULTICLASS_RECOMMEND, list)
    assert isinstance(IMBALANCED_RECOMMEND, list)
    assert "ACC" in BINARY_RECOMMEND
    assert "Overall ACC" in MULTICLASS_RECOMMEND
    assert "Kappa" in IMBALANCED_RECOMMEND


def test_ndtri_coefficients():
    """Test NDTRI polynomial coefficients are defined."""
    from pycm.params import NDTRI_P0, NDTRI_Q0, NDTRI_P1, NDTRI_Q1, NDTRI_P2, NDTRI_Q2
    assert isinstance(NDTRI_P0, list)
    assert isinstance(NDTRI_Q0, list)
    assert isinstance(NDTRI_P1, list)
    assert isinstance(NDTRI_Q1, list)
    assert isinstance(NDTRI_P2, list)
    assert isinstance(NDTRI_Q2, list)
    assert len(NDTRI_P0) == 5
    assert len(NDTRI_Q0) == 8


def test_benchmark_color():
    """Test BENCHMARK_COLOR dictionary."""
    from pycm.params import BENCHMARK_COLOR, BENCHMARK_LIST
    assert isinstance(BENCHMARK_COLOR, dict)
    assert BENCHMARK_LIST == list(BENCHMARK_COLOR)
    assert "PLRI" in BENCHMARK_COLOR
    assert BENCHMARK_COLOR["PLRI"]["Good"] == "Green"
    assert "SOA1(Landis & Koch)" in BENCHMARK_COLOR
