# -*- coding: utf-8 -*-
"""
Tests for the interpret module.
"""
import pytest
import math


def test_Q_analysis():
    """Test Q_analysis function (Yule's Q interpretation)."""
    from pycm.interpret import Q_analysis

    # Test Negligible range
    assert Q_analysis(0) == "Negligible"
    assert Q_analysis(0.1) == "Negligible"
    assert Q_analysis(0.24) == "Negligible"

    # Test Weak range
    assert Q_analysis(0.25) == "Weak"
    assert Q_analysis(0.3) == "Weak"
    assert Q_analysis(0.49) == "Weak"

    # Test Moderate range
    assert Q_analysis(0.5) == "Moderate"
    assert Q_analysis(0.6) == "Moderate"
    assert Q_analysis(0.74) == "Moderate"

    # Test Strong range
    assert Q_analysis(0.75) == "Strong"
    assert Q_analysis(0.9) == "Strong"
    assert Q_analysis(1.0) == "Strong"

    # Test None cases
    assert Q_analysis(float('nan')) == "None"
    assert Q_analysis(None) == "None"
    assert Q_analysis("None") == "None"


def test_MCC_analysis():
    """Test MCC_analysis function (Matthews correlation coefficient)."""
    from pycm.interpret import MCC_analysis

    # Test Negligible
    assert MCC_analysis(0.1) == "Negligible"
    assert MCC_analysis(0.29) == "Negligible"

    # Test Weak
    assert MCC_analysis(0.3) == "Weak"
    assert MCC_analysis(0.49) == "Weak"

    # Test Moderate
    assert MCC_analysis(0.5) == "Moderate"
    assert MCC_analysis(0.69) == "Moderate"

    # Test Strong
    assert MCC_analysis(0.7) == "Strong"
    assert MCC_analysis(0.89) == "Strong"

    # Test Very Strong
    assert MCC_analysis(0.9) == "Very Strong"
    assert MCC_analysis(1.0) == "Very Strong"

    # Test None cases
    assert MCC_analysis(float('nan')) == "None"


def test_NLR_analysis():
    """Test NLR_analysis function (Negative likelihood ratio)."""
    from pycm.interpret import NLR_analysis

    # Test Good
    assert NLR_analysis(0.05) == "Good"
    assert NLR_analysis(0.09) == "Good"

    # Test Fair
    assert NLR_analysis(0.1) == "Fair"
    assert NLR_analysis(0.19) == "Fair"

    # Test Poor
    assert NLR_analysis(0.2) == "Poor"
    assert NLR_analysis(0.49) == "Poor"

    # Test Negligible
    assert NLR_analysis(0.5) == "Negligible"
    assert NLR_analysis(1.0) == "Negligible"

    # Test None cases
    assert NLR_analysis(float('nan')) == "None"


def test_V_analysis():
    """Test V_analysis function (Cramer's V)."""
    from pycm.interpret import V_analysis

    # Test Negligible
    assert V_analysis(0.05) == "Negligible"
    assert V_analysis(0.09) == "Negligible"

    # Test Weak
    assert V_analysis(0.1) == "Weak"
    assert V_analysis(0.19) == "Weak"

    # Test Moderate
    assert V_analysis(0.2) == "Moderate"
    assert V_analysis(0.39) == "Moderate"

    # Test Relatively Strong
    assert V_analysis(0.4) == "Relatively Strong"
    assert V_analysis(0.59) == "Relatively Strong"

    # Test Strong
    assert V_analysis(0.6) == "Strong"
    assert V_analysis(0.79) == "Strong"

    # Test Very Strong
    assert V_analysis(0.8) == "Very Strong"
    assert V_analysis(1.0) == "Very Strong"

    # Test None cases
    assert V_analysis(float('nan')) == "None"


def test_PLR_analysis():
    """Test PLR_analysis function (Positive likelihood ratio)."""
    from pycm.interpret import PLR_analysis

    # Test Negligible
    assert PLR_analysis(0.5) == "Negligible"
    assert PLR_analysis(0.9) == "Negligible"

    # Test Poor
    assert PLR_analysis(1) == "Poor"
    assert PLR_analysis(3) == "Poor"
    assert PLR_analysis(4.9) == "Poor"

    # Test Fair
    assert PLR_analysis(5) == "Fair"
    assert PLR_analysis(7) == "Fair"
    assert PLR_analysis(9.9) == "Fair"

    # Test Good
    assert PLR_analysis(10) == "Good"
    assert PLR_analysis(11) == "Good"
    assert PLR_analysis(100) == "Good"

    # Test None cases
    assert PLR_analysis("None") == "None"
    assert PLR_analysis(float('nan')) == "None"


def test_DP_analysis():
    """Test DP_analysis function (Discriminant power)."""
    from pycm.interpret import DP_analysis

    # Test Poor
    assert DP_analysis(0.2) == "Poor"
    assert DP_analysis(0.9) == "Poor"

    # Test Limited
    assert DP_analysis(1) == "Limited"
    assert DP_analysis(1.5) == "Limited"
    assert DP_analysis(1.9) == "Limited"

    # Test Fair
    assert DP_analysis(2) == "Fair"
    assert DP_analysis(2.5) == "Fair"
    assert DP_analysis(2.9) == "Fair"

    # Test Good
    assert DP_analysis(3) == "Good"
    assert DP_analysis(10) == "Good"

    # Test None cases
    assert DP_analysis(float('nan')) == "None"


def test_AUC_analysis():
    """Test AUC_analysis function (Area under ROC curve)."""
    from pycm.interpret import AUC_analysis

    # Test Poor
    assert AUC_analysis(0.5) == "Poor"
    assert AUC_analysis(0.59) == "Poor"

    # Test Fair
    assert AUC_analysis(0.6) == "Fair"
    assert AUC_analysis(0.65) == "Fair"
    assert AUC_analysis(0.69) == "Fair"

    # Test Good
    assert AUC_analysis(0.7) == "Good"
    assert AUC_analysis(0.75) == "Good"
    assert AUC_analysis(0.79) == "Good"

    # Test Very Good
    assert AUC_analysis(0.8) == "Very Good"
    assert AUC_analysis(0.86) == "Very Good"
    assert AUC_analysis(0.89) == "Very Good"

    # Test Excellent
    assert AUC_analysis(0.9) == "Excellent"
    assert AUC_analysis(0.97) == "Excellent"
    assert AUC_analysis(1.0) == "Excellent"

    # Test None cases
    assert AUC_analysis(float('nan')) == "None"


def test_kappa_analysis_cicchetti():
    """Test kappa_analysis_cicchetti function."""
    from pycm.interpret import kappa_analysis_cicchetti

    # Test Poor
    assert kappa_analysis_cicchetti(0.3) == "Poor"
    assert kappa_analysis_cicchetti(0.39) == "Poor"

    # Test Fair
    assert kappa_analysis_cicchetti(0.4) == "Fair"
    assert kappa_analysis_cicchetti(0.5) == "Fair"
    assert kappa_analysis_cicchetti(0.58) == "Fair"

    # Test Good
    assert kappa_analysis_cicchetti(0.59) == "Good"
    assert kappa_analysis_cicchetti(0.65) == "Good"
    assert kappa_analysis_cicchetti(0.73) == "Good"

    # Test Excellent
    assert kappa_analysis_cicchetti(0.74) == "Excellent"
    assert kappa_analysis_cicchetti(0.8) == "Excellent"
    assert kappa_analysis_cicchetti(1.0) == "Excellent"

    # Test None cases
    assert kappa_analysis_cicchetti(1.2) == "None"
    assert kappa_analysis_cicchetti(float('nan')) == "None"


def test_kappa_analysis_koch():
    """Test kappa_analysis_koch function (Landis-Koch benchmark)."""
    from pycm.interpret import kappa_analysis_koch

    # Test Poor
    assert kappa_analysis_koch(-0.1) == "Poor"
    assert kappa_analysis_koch(-0.5) == "Poor"

    # Test Slight
    assert kappa_analysis_koch(0) == "Slight"
    assert kappa_analysis_koch(0.1) == "Slight"
    assert kappa_analysis_koch(0.19) == "Slight"

    # Test Fair
    assert kappa_analysis_koch(0.2) == "Fair"
    assert kappa_analysis_koch(0.3) == "Fair"
    assert kappa_analysis_koch(0.39) == "Fair"

    # Test Moderate
    assert kappa_analysis_koch(0.4) == "Moderate"
    assert kappa_analysis_koch(0.5) == "Moderate"
    assert kappa_analysis_koch(0.59) == "Moderate"

    # Test Substantial
    assert kappa_analysis_koch(0.6) == "Substantial"
    assert kappa_analysis_koch(0.7) == "Substantial"
    assert kappa_analysis_koch(0.79) == "Substantial"

    # Test Almost Perfect
    assert kappa_analysis_koch(0.8) == "Almost Perfect"
    assert kappa_analysis_koch(0.9) == "Almost Perfect"
    assert kappa_analysis_koch(1.0) == "Almost Perfect"

    # Test None cases
    assert kappa_analysis_koch(1.2) == "None"
    assert kappa_analysis_koch(float('nan')) == "None"


def test_kappa_analysis_fleiss():
    """Test kappa_analysis_fleiss function."""
    from pycm.interpret import kappa_analysis_fleiss

    # Test Poor
    assert kappa_analysis_fleiss(0.2) == "Poor"
    assert kappa_analysis_fleiss(0.39) == "Poor"

    # Test Intermediate to Good
    assert kappa_analysis_fleiss(0.4) == "Intermediate to Good"
    assert kappa_analysis_fleiss(0.6) == "Intermediate to Good"
    assert kappa_analysis_fleiss(0.74) == "Intermediate to Good"

    # Test Excellent
    assert kappa_analysis_fleiss(0.75) == "Excellent"
    assert kappa_analysis_fleiss(1.0) == "Excellent"
    assert kappa_analysis_fleiss(1.2) == "Excellent"  # Outside normal range but still Excellent

    # Test None cases
    assert kappa_analysis_fleiss(float('nan')) == "None"


def test_kappa_analysis_altman():
    """Test kappa_analysis_altman function."""
    from pycm.interpret import kappa_analysis_altman

    # Test Poor
    assert kappa_analysis_altman(-0.2) == "Poor"
    assert kappa_analysis_altman(0.1) == "Poor"
    assert kappa_analysis_altman(0.19) == "Poor"

    # Test Fair
    assert kappa_analysis_altman(0.2) == "Fair"
    assert kappa_analysis_altman(0.3) == "Fair"
    assert kappa_analysis_altman(0.39) == "Fair"

    # Test Moderate
    assert kappa_analysis_altman(0.4) == "Moderate"
    assert kappa_analysis_altman(0.5) == "Moderate"
    assert kappa_analysis_altman(0.59) == "Moderate"

    # Test Good
    assert kappa_analysis_altman(0.6) == "Good"
    assert kappa_analysis_altman(0.7) == "Good"
    assert kappa_analysis_altman(0.79) == "Good"

    # Test Very Good
    assert kappa_analysis_altman(0.8) == "Very Good"
    assert kappa_analysis_altman(0.9) == "Very Good"
    assert kappa_analysis_altman(1.0) == "Very Good"

    # Test None cases
    assert kappa_analysis_altman(1.2) == "None"
    assert kappa_analysis_altman(float('nan')) == "None"


def test_lambda_analysis():
    """Test lambda_analysis function."""
    from pycm.interpret import lambda_analysis

    # Test None (lambda = 0)
    assert lambda_analysis(0) == "None"

    # Test Very Weak
    assert lambda_analysis(0.1) == "Very Weak"
    assert lambda_analysis(0.19) == "Very Weak"

    # Test Weak
    assert lambda_analysis(0.2) == "Weak"
    assert lambda_analysis(0.3) == "Weak"
    assert lambda_analysis(0.39) == "Weak"

    # Test Moderate
    assert lambda_analysis(0.4) == "Moderate"
    assert lambda_analysis(0.5) == "Moderate"
    assert lambda_analysis(0.59) == "Moderate"

    # Test Strong
    assert lambda_analysis(0.6) == "Strong"
    assert lambda_analysis(0.7) == "Strong"
    assert lambda_analysis(0.79) == "Strong"

    # Test Very Strong
    assert lambda_analysis(0.8) == "Very Strong"
    assert lambda_analysis(0.9) == "Very Strong"
    assert lambda_analysis(0.99) == "Very Strong"

    # Test Perfect
    assert lambda_analysis(1) == "Perfect"
    assert lambda_analysis(1.0) == "Perfect"

    # Test None cases
    assert lambda_analysis(float('nan')) == "None"


def test_alpha_analysis():
    """Test alpha_analysis function (Krippendorff's alpha)."""
    from pycm.interpret import alpha_analysis

    # Test Low
    assert alpha_analysis(0) == "Low"
    assert alpha_analysis(0.5) == "Low"
    assert alpha_analysis(0.666) == "Low"

    # Test Tentative
    assert alpha_analysis(0.667) == "Tentative"
    assert alpha_analysis(0.7) == "Tentative"
    assert alpha_analysis(0.799) == "Tentative"

    # Test High
    assert alpha_analysis(0.8) == "High"
    assert alpha_analysis(0.9) == "High"
    assert alpha_analysis(1.0) == "High"

    # Test None cases
    assert alpha_analysis(float('nan')) == "None"


def test_pearson_C_analysis():
    """Test pearson_C_analysis function."""
    from pycm.interpret import pearson_C_analysis

    # Test None (pearson_C = 0)
    assert pearson_C_analysis(0) == "None"

    # Test Not Appreciable
    assert pearson_C_analysis(0.05) == "Not Appreciable"
    assert pearson_C_analysis(0.099) == "Not Appreciable"

    # Test Weak
    assert pearson_C_analysis(0.1) == "Weak"
    assert pearson_C_analysis(0.15) == "Weak"
    assert pearson_C_analysis(0.199) == "Weak"

    # Test Medium
    assert pearson_C_analysis(0.2) == "Medium"
    assert pearson_C_analysis(0.25) == "Medium"
    assert pearson_C_analysis(0.299) == "Medium"

    # Test Strong
    assert pearson_C_analysis(0.3) == "Strong"
    assert pearson_C_analysis(0.5) == "Strong"
    assert pearson_C_analysis(1.0) == "Strong"

    # Test None cases
    assert pearson_C_analysis(float('nan')) == "None"


def test_all_functions_exist():
    """Test that all interpretation functions are importable."""
    from pycm.interpret import (
        Q_analysis, MCC_analysis, NLR_analysis, V_analysis,
        PLR_analysis, DP_analysis, AUC_analysis,
        kappa_analysis_cicchetti, kappa_analysis_koch,
        kappa_analysis_fleiss, kappa_analysis_altman,
        lambda_analysis, alpha_analysis, pearson_C_analysis
    )
    # If we get here, all functions exist
    assert True
