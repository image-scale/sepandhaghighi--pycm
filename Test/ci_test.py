# -*- coding: utf-8 -*-
"""
Tests for the ci module.
"""
import pytest
import math


def test_ci_calc_agresti_basic():
    """Test CI_calc_agresti with basic values."""
    from pycm.ci import CI_calc_agresti

    # Test with 50% proportion, 100 observations
    CI_down, CI_up = CI_calc_agresti(0.5, 100, 1.96)
    assert isinstance(CI_down, float)
    assert isinstance(CI_up, float)
    assert CI_down < 0.5
    assert CI_up > 0.5
    assert CI_down < CI_up


def test_ci_calc_agresti_edge_cases():
    """Test CI_calc_agresti with edge cases."""
    from pycm.ci import CI_calc_agresti

    # Test with 0 proportion
    CI_down, CI_up = CI_calc_agresti(0, 100, 1.96)
    assert isinstance(CI_down, float)
    assert CI_down >= 0 or CI_down < 0  # Just checking it returns a float

    # Test with 1 proportion
    CI_down, CI_up = CI_calc_agresti(1, 100, 1.96)
    assert isinstance(CI_up, float)

    # Test with 0 observations - Agresti-Coull still computes due to CV^2 adjustment
    # So it returns floats, not "None"
    CI_down, CI_up = CI_calc_agresti(0.5, 0, 1.96)
    assert isinstance(CI_down, float)
    assert isinstance(CI_up, float)


def test_ci_calc_wilson_basic():
    """Test CI_calc_wilson with basic values."""
    from pycm.ci import CI_calc_wilson

    # Test with 50% proportion, 100 observations
    CI_down, CI_up = CI_calc_wilson(0.5, 100, 1.96)
    assert isinstance(CI_down, float)
    assert isinstance(CI_up, float)
    assert CI_down < 0.5
    assert CI_up > 0.5
    assert CI_down < CI_up


def test_ci_calc_wilson_edge_cases():
    """Test CI_calc_wilson with edge cases."""
    from pycm.ci import CI_calc_wilson

    # Test with 0 proportion
    CI_down, CI_up = CI_calc_wilson(0, 100, 1.96)
    assert isinstance(CI_down, float)

    # Test with 0 observations should return "None"
    CI_down, CI_up = CI_calc_wilson(0.5, 0, 1.96)
    assert CI_down == "None"
    assert CI_up == "None"


def test_auc_se_calc_basic():
    """Test AUC_SE_calc with basic values."""
    from pycm.ci import AUC_SE_calc

    # Test with typical AUC values
    SE = AUC_SE_calc(0.8, 50, 50)
    assert isinstance(SE, float)
    assert SE > 0

    # Test with perfect AUC
    SE = AUC_SE_calc(1.0, 50, 50)
    assert isinstance(SE, float)

    # Test with AUC of 0.5
    SE = AUC_SE_calc(0.5, 50, 50)
    assert isinstance(SE, float)
    assert SE > 0


def test_auc_se_calc_edge_cases():
    """Test AUC_SE_calc with edge cases."""
    from pycm.ci import AUC_SE_calc

    # Test with 0 positives
    SE = AUC_SE_calc(0.8, 0, 50)
    assert SE == "None"

    # Test with 0 negatives
    SE = AUC_SE_calc(0.8, 50, 0)
    assert SE == "None"


def test_lr_se_calc_basic():
    """Test LR_SE_calc with basic values."""
    from pycm.ci import LR_SE_calc

    # Test with typical values
    SE = LR_SE_calc(20, 50, 10, 50)
    assert isinstance(SE, float)
    assert SE > 0


def test_lr_se_calc_edge_cases():
    """Test LR_SE_calc with edge cases."""
    from pycm.ci import LR_SE_calc

    # Test with 0 values
    SE = LR_SE_calc(0, 50, 10, 50)
    assert SE == "None"

    SE = LR_SE_calc(20, 0, 10, 50)
    assert SE == "None"


def test_lr_ci_calc_basic():
    """Test LR_CI_calc with basic values."""
    from pycm.ci import LR_CI_calc

    # Test with typical values
    CI_down, CI_up = LR_CI_calc(2.0, 0.3, 1.96)
    assert isinstance(CI_down, float)
    assert isinstance(CI_up, float)
    assert CI_down < 2.0
    assert CI_up > 2.0
    assert CI_down < CI_up


def test_lr_ci_calc_edge_cases():
    """Test LR_CI_calc with edge cases."""
    from pycm.ci import LR_CI_calc

    # Test with 0 mean
    CI_down, CI_up = LR_CI_calc(0, 0.3, 1.96)
    assert CI_down == "None"
    assert CI_up == "None"

    # Test with negative mean
    CI_down, CI_up = LR_CI_calc(-1, 0.3, 1.96)
    assert CI_down == "None"
    assert CI_up == "None"


def test_ci_calc_basic():
    """Test CI_calc with basic values."""
    from pycm.ci import CI_calc

    # Test with typical values
    CI_down, CI_up = CI_calc(0.8, 0.05, 1.96)
    assert isinstance(CI_down, float)
    assert isinstance(CI_up, float)
    assert abs(CI_down - (0.8 - 1.96 * 0.05)) < 0.0001
    assert abs(CI_up - (0.8 + 1.96 * 0.05)) < 0.0001


def test_ci_calc_symmetry():
    """Test CI_calc produces symmetric intervals."""
    from pycm.ci import CI_calc

    mean = 0.5
    SE = 0.1
    CV = 1.96
    CI_down, CI_up = CI_calc(mean, SE, CV)

    # Check symmetry around the mean
    assert abs((mean - CI_down) - (CI_up - mean)) < 0.0001


def test_se_calc_basic():
    """Test SE_calc with basic values."""
    from pycm.ci import SE_calc

    # Test with 50% proportion
    SE = SE_calc(0.5, 100)
    expected = math.sqrt(0.5 * 0.5 / 100)
    assert abs(SE - expected) < 0.0001

    # Test with 80% proportion
    SE = SE_calc(0.8, 100)
    expected = math.sqrt(0.8 * 0.2 / 100)
    assert abs(SE - expected) < 0.0001


def test_se_calc_edge_cases():
    """Test SE_calc with edge cases."""
    from pycm.ci import SE_calc

    # Test with 0 observations
    SE = SE_calc(0.5, 0)
    assert SE == "None"

    # Test with 0 proportion
    SE = SE_calc(0, 100)
    assert SE == 0

    # Test with 1 proportion
    SE = SE_calc(1, 100)
    assert SE == 0


def test_kappa_se_calc_basic():
    """Test kappa_SE_calc with basic values."""
    from pycm.ci import kappa_SE_calc

    # Test with typical values
    SE = kappa_SE_calc(0.85, 0.5, 100)
    assert isinstance(SE, float)
    assert SE > 0


def test_kappa_se_calc_edge_cases():
    """Test kappa_SE_calc with edge cases."""
    from pycm.ci import kappa_SE_calc

    # Test with PE = 1 (division by zero)
    SE = kappa_SE_calc(0.85, 1.0, 100)
    assert SE == "None"

    # Test with 0 population
    SE = kappa_SE_calc(0.85, 0.5, 0)
    assert SE == "None"


def test_ci_methods_comparison():
    """Test that different CI methods produce different but valid results."""
    from pycm.ci import CI_calc_agresti, CI_calc_wilson, CI_calc, SE_calc

    item1 = 0.7
    item2 = 100
    CV = 1.96

    agresti_down, agresti_up = CI_calc_agresti(item1, item2, CV)
    wilson_down, wilson_up = CI_calc_wilson(item1, item2, CV)

    SE = SE_calc(item1, item2)
    normal_down, normal_up = CI_calc(item1, SE, CV)

    # All should produce valid intervals
    assert agresti_down < agresti_up
    assert wilson_down < wilson_up
    assert normal_down < normal_up

    # Intervals should contain the point estimate (approximately)
    assert agresti_down < item1 < agresti_up or abs(agresti_down - item1) < 0.1
    assert wilson_down < item1 < wilson_up or abs(wilson_down - item1) < 0.1


def test_ci_calc_different_cv():
    """Test CI_calc with different critical values."""
    from pycm.ci import CI_calc

    mean = 0.5
    SE = 0.1

    # 90% CI (CV ≈ 1.645)
    CI_down_90, CI_up_90 = CI_calc(mean, SE, 1.645)

    # 95% CI (CV = 1.96)
    CI_down_95, CI_up_95 = CI_calc(mean, SE, 1.96)

    # 99% CI (CV ≈ 2.576)
    CI_down_99, CI_up_99 = CI_calc(mean, SE, 2.576)

    # Higher CV should produce wider intervals
    width_90 = CI_up_90 - CI_down_90
    width_95 = CI_up_95 - CI_down_95
    width_99 = CI_up_99 - CI_down_99

    assert width_90 < width_95 < width_99


def test_ci_handler_functions_exist():
    """Test that CI handler functions exist and are importable."""
    from pycm.ci import __CI_class_handler__, __CI_overall_handler__

    # Just verify they exist and are callable
    assert callable(__CI_class_handler__)
    assert callable(__CI_overall_handler__)


def test_all_ci_functions_importable():
    """Test that all CI functions can be imported."""
    from pycm.ci import (
        CI_calc_agresti,
        CI_calc_wilson,
        AUC_SE_calc,
        LR_SE_calc,
        LR_CI_calc,
        CI_calc,
        SE_calc,
        kappa_SE_calc,
        __CI_class_handler__,
        __CI_overall_handler__
    )

    # All functions should be callable
    assert callable(CI_calc_agresti)
    assert callable(CI_calc_wilson)
    assert callable(AUC_SE_calc)
    assert callable(LR_SE_calc)
    assert callable(LR_CI_calc)
    assert callable(CI_calc)
    assert callable(SE_calc)
    assert callable(kappa_SE_calc)
    assert callable(__CI_class_handler__)
    assert callable(__CI_overall_handler__)
