# -*- coding: utf-8 -*-
"""
Tests for the overall_funcs module.
"""
import pytest
import math


def test_reliability_calc():
    """Test reliability_calc function."""
    from pycm.overall_funcs import reliability_calc

    # Kappa calculation
    assert abs(reliability_calc(0.25, 0.85) - 0.8) < 0.0001
    assert reliability_calc(1.0, 0.85) == "None"  # Division by zero
    assert reliability_calc(0.5, 0.5) == 0.0


def test_overall_accuracy_calc():
    """Test overall_accuracy_calc function."""
    from pycm.overall_funcs import overall_accuracy_calc

    TP = {0: 50, 1: 35}
    assert overall_accuracy_calc(TP, 100) == 0.85

    assert overall_accuracy_calc({0: 0}, 0) == "None"


def test_overall_random_accuracy_calc():
    """Test overall_random_accuracy_calc function."""
    from pycm.overall_funcs import overall_random_accuracy_calc

    RACC = {0: 0.25, 1: 0.25}
    assert overall_random_accuracy_calc(RACC) == 0.5


def test_micro_calc():
    """Test micro_calc function."""
    from pycm.overall_funcs import micro_calc

    TP = {0: 50, 1: 35}
    FN = {0: 10, 1: 5}
    # TPR_micro = (50+35) / (50+35+10+5) = 85/100 = 0.85
    assert micro_calc(TP, FN) == 0.85


def test_macro_calc():
    """Test macro_calc function."""
    from pycm.overall_funcs import macro_calc

    TPR = {0: 0.8, 1: 0.9}
    assert abs(macro_calc(TPR) - 0.85) < 0.0001


def test_ncr():
    """Test ncr function (combinations)."""
    from pycm.overall_funcs import ncr

    assert ncr(5, 2) == 10
    assert ncr(10, 3) == 120
    assert ncr(5, 5) == 1
    assert ncr(5, 0) == 1
    assert ncr(3, 5) == 0  # r > n


def test_chi_square_calc():
    """Test chi_square_calc function."""
    from pycm.overall_funcs import chi_square_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    TOP = {0: 55, 1: 45}
    P = {0: 60, 1: 40}
    POP = {0: 100, 1: 100}

    result = chi_square_calc(classes, table, TOP, P, POP)
    assert isinstance(result, float)
    assert result > 0


def test_phi_square_calc():
    """Test phi_square_calc function."""
    from pycm.overall_funcs import phi_square_calc

    assert phi_square_calc(10.0, 100) == 0.1
    assert phi_square_calc(50.0, 0) == "None"


def test_cramers_v_calc():
    """Test cramers_V_calc function."""
    from pycm.overall_funcs import cramers_V_calc

    result = cramers_V_calc(0.25, [0, 1])
    assert abs(result - 0.5) < 0.0001


def test_df_calc():
    """Test DF_calc function."""
    from pycm.overall_funcs import DF_calc

    assert DF_calc([0, 1]) == 1
    assert DF_calc([0, 1, 2]) == 4


def test_entropy_calc():
    """Test entropy_calc function."""
    from pycm.overall_funcs import entropy_calc

    TOP = {0: 50, 1: 50}
    POP = {0: 100, 1: 100}

    result = entropy_calc(TOP, POP)
    assert isinstance(result, float)
    assert result >= 0


def test_joint_entropy_calc():
    """Test joint_entropy_calc function."""
    from pycm.overall_funcs import joint_entropy_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    POP = {0: 100, 1: 100}

    result = joint_entropy_calc(classes, table, POP)
    assert isinstance(result, float)
    assert result >= 0


def test_conditional_entropy_calc():
    """Test conditional_entropy_calc function."""
    from pycm.overall_funcs import conditional_entropy_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    P = {0: 60, 1: 40}
    POP = {0: 100, 1: 100}

    result = conditional_entropy_calc(classes, table, P, POP)
    assert isinstance(result, float)
    assert result >= 0


def test_mutual_information_calc():
    """Test mutual_information_calc function."""
    from pycm.overall_funcs import mutual_information_calc

    assert mutual_information_calc(1.5, 0.5) == 1.0
    assert mutual_information_calc("None", 0.5) == "None"


def test_kl_divergence_calc():
    """Test kl_divergence_calc function."""
    from pycm.overall_funcs import kl_divergence_calc

    P = {0: 50, 1: 50}
    TOP = {0: 55, 1: 45}
    POP = {0: 100, 1: 100}

    result = kl_divergence_calc(P, TOP, POP)
    assert isinstance(result, float)


def test_cross_entropy_calc():
    """Test cross_entropy_calc function."""
    from pycm.overall_funcs import cross_entropy_calc

    TOP = {0: 50, 1: 50}
    P = {0: 60, 1: 40}
    POP = {0: 100, 1: 100}

    result = cross_entropy_calc(TOP, P, POP)
    assert isinstance(result, float)


def test_lambda_a_calc():
    """Test lambda_A_calc function."""
    from pycm.overall_funcs import lambda_A_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    P = {0: 60, 1: 40}

    result = lambda_A_calc(classes, table, P, 100)
    assert isinstance(result, float)


def test_lambda_b_calc():
    """Test lambda_B_calc function."""
    from pycm.overall_funcs import lambda_B_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    TOP = {0: 55, 1: 45}

    result = lambda_B_calc(classes, table, TOP, 100)
    assert isinstance(result, float)


def test_hamming_calc():
    """Test hamming_calc function."""
    from pycm.overall_funcs import hamming_calc

    TP = {0: 50, 1: 35}
    result = hamming_calc(TP, 100)
    assert abs(result - 0.15) < 0.0001


def test_zero_one_loss_calc():
    """Test zero_one_loss_calc function."""
    from pycm.overall_funcs import zero_one_loss_calc

    TP = {0: 50, 1: 35}
    result = zero_one_loss_calc(TP, 100)
    assert result == 15


def test_nir_calc():
    """Test NIR_calc function."""
    from pycm.overall_funcs import NIR_calc

    P = {0: 60, 1: 40}
    result = NIR_calc(P, 100)
    assert result == 0.6


def test_p_value_calc():
    """Test p_value_calc function."""
    from pycm.overall_funcs import p_value_calc

    TP = {0: 50, 1: 35}
    result = p_value_calc(TP, 100, 0.6)
    assert isinstance(result, float)


def test_overall_jaccard_index_calc():
    """Test overall_jaccard_index_calc function."""
    from pycm.overall_funcs import overall_jaccard_index_calc

    jaccard_list = [0.8, 0.7, 0.75]
    result = overall_jaccard_index_calc(jaccard_list)
    assert isinstance(result, tuple)
    assert abs(result[0] - 2.25) < 0.0001
    assert abs(result[1] - 0.75) < 0.0001


def test_kappa_no_prevalence_calc():
    """Test kappa_no_prevalence_calc function."""
    from pycm.overall_funcs import kappa_no_prevalence_calc

    assert kappa_no_prevalence_calc(0.85) == 0.7
    assert kappa_no_prevalence_calc("None") == "None"


def test_pc_s_calc():
    """Test PC_S_calc function."""
    from pycm.overall_funcs import PC_S_calc

    assert PC_S_calc([0, 1]) == 0.5
    assert abs(PC_S_calc([0, 1, 2]) - 1/3) < 0.0001


def test_pc_ac1_calc():
    """Test PC_AC1_calc function."""
    from pycm.overall_funcs import PC_AC1_calc

    P = {0: 60, 1: 40}
    TOP = {0: 55, 1: 45}
    POP = {0: 100, 1: 100}

    result = PC_AC1_calc(P, TOP, POP)
    assert isinstance(result, float)


def test_rr_calc():
    """Test RR_calc function."""
    from pycm.overall_funcs import RR_calc

    classes = [0, 1]
    TOP = {0: 55, 1: 45}
    result = RR_calc(classes, TOP)
    assert result == 50.0


def test_cba_calc():
    """Test CBA_calc function."""
    from pycm.overall_funcs import CBA_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    TOP = {0: 55, 1: 45}
    P = {0: 60, 1: 40}

    result = CBA_calc(classes, table, TOP, P)
    assert isinstance(result, float)


def test_overall_mcc_calc():
    """Test overall_MCC_calc function."""
    from pycm.overall_funcs import overall_MCC_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    TOP = {0: 55, 1: 45}
    P = {0: 60, 1: 40}

    result = overall_MCC_calc(classes, table, TOP, P)
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_overall_cen_calc():
    """Test overall_CEN_calc function."""
    from pycm.overall_funcs import overall_CEN_calc

    classes = [0, 1]
    TP = {0: 50, 1: 35}
    TOP = {0: 55, 1: 45}
    P = {0: 60, 1: 40}
    CEN_dict = {0: 0.3, 1: 0.4}

    result = overall_CEN_calc(classes, TP, TOP, P, CEN_dict)
    assert isinstance(result, float)


def test_convex_combination():
    """Test convex_combination function."""
    from pycm.overall_funcs import convex_combination

    classes = [0, 1]
    TP = {0: 50, 1: 35}
    TOP = {0: 55, 1: 45}
    P = {0: 60, 1: 40}

    result = convex_combination(classes, TP, TOP, P, 0)
    assert isinstance(result, float)


def test_b_calc():
    """Test B_calc function (Bangdiwala's B)."""
    from pycm.overall_funcs import B_calc

    classes = [0, 1]
    TP = {0: 50, 1: 35}
    TOP = {0: 55, 1: 45}
    P = {0: 60, 1: 40}

    result = B_calc(classes, TP, TOP, P)
    assert isinstance(result, float)


def test_ari_calc():
    """Test ARI_calc function."""
    from pycm.overall_funcs import ARI_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    TOP = {0: 55, 1: 45}
    P = {0: 60, 1: 40}

    result = ARI_calc(classes, table, TOP, P, 100)
    assert isinstance(result, float)


def test_alpha_calc():
    """Test alpha_calc function (Krippendorff's alpha)."""
    from pycm.overall_funcs import alpha_calc

    result = alpha_calc(0.5, 0.85, 100)
    assert isinstance(result, float)


def test_pearson_c_calc():
    """Test pearson_C_calc function."""
    from pycm.overall_funcs import pearson_C_calc

    result = pearson_C_calc(25.0, 100)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_rci_calc():
    """Test RCI_calc function."""
    from pycm.overall_funcs import RCI_calc

    assert RCI_calc(0.5, 1.0) == 0.5
    assert RCI_calc(0.5, 0) == "None"


def test_aunp_calc():
    """Test AUNP_calc function."""
    from pycm.overall_funcs import AUNP_calc

    classes = [0, 1]
    P = {0: 60, 1: 40}
    POP = {0: 100, 1: 100}
    AUC_dict = {0: 0.9, 1: 0.85}

    result = AUNP_calc(classes, P, POP, AUC_dict)
    assert isinstance(result, float)


def test_brier_score_calc():
    """Test brier_score_calc function."""
    from pycm.overall_funcs import brier_score_calc

    classes = [0, 1]
    prob_vector = [0.8, 0.6, 0.3, 0.7]
    actual_vector = [1, 1, 0, 1]

    result = brier_score_calc(classes, prob_vector, actual_vector)
    assert isinstance(result, float)


def test_log_loss_calc():
    """Test log_loss_calc function."""
    from pycm.overall_funcs import log_loss_calc

    classes = [0, 1]
    prob_vector = [0.8, 0.6, 0.3, 0.7]
    actual_vector = [1, 1, 0, 1]

    result = log_loss_calc(classes, prob_vector, actual_vector)
    assert isinstance(result, float)


def test_weighted_kappa_calc():
    """Test weighted_kappa_calc function."""
    from pycm.overall_funcs import weighted_kappa_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    P = {0: 60, 1: 40}
    TOP = {0: 55, 1: 45}
    POP = {0: 100, 1: 100}
    weight = {0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}}

    result = weighted_kappa_calc(classes, table, P, TOP, POP, weight)
    assert isinstance(result, float)


def test_overall_statistics():
    """Test overall_statistics function."""
    from pycm.overall_funcs import overall_statistics

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}
    TP = {0: 50, 1: 35}
    TN = {0: 35, 1: 50}
    FP = {0: 5, 1: 10}
    FN = {0: 10, 1: 5}
    P = {0: 60, 1: 40}
    TOP = {0: 55, 1: 45}
    POP = {0: 100, 1: 100}
    RACC = {0: 0.33, 1: 0.18}
    RACCU = {0: 0.33, 1: 0.18}
    jaccard_list = {0: 0.769, 1: 0.7}
    CEN_dict = {0: 0.3, 1: 0.4}
    MCEN_dict = {0: 0.35, 1: 0.45}
    AUC_dict = {0: 0.9, 1: 0.85}
    TPR = {0: 0.833, 1: 0.875}
    TNR = {0: 0.875, 1: 0.833}
    PPV = {0: 0.909, 1: 0.777}
    NPV = {0: 0.777, 1: 0.909}
    ACC = {0: 0.85, 1: 0.85}
    F1 = {0: 0.87, 1: 0.82}
    ICSI_dict = {0: 0.74, 1: 0.65}

    result = overall_statistics(
        classes=classes,
        table=table,
        TP=TP,
        TN=TN,
        FP=FP,
        FN=FN,
        P=P,
        TOP=TOP,
        POP=POP,
        RACC=RACC,
        RACCU=RACCU,
        jaccard_list=jaccard_list,
        CEN_dict=CEN_dict,
        MCEN_dict=MCEN_dict,
        AUC_dict=AUC_dict,
        TPR=TPR,
        TNR=TNR,
        PPV=PPV,
        NPV=NPV,
        ACC=ACC,
        F1=F1,
        ICSI_dict=ICSI_dict
    )

    # Check key statistics exist
    assert "Overall ACC" in result
    assert "Kappa" in result
    assert "Overall MCC" in result
    assert "Chi-Squared" in result
    assert "Response Entropy" in result
    assert "Reference Entropy" in result
    assert "Hamming Loss" in result
    assert "TPR Macro" in result
    assert "TPR Micro" in result

    # Check types
    assert isinstance(result["Overall ACC"], float)
    assert isinstance(result["Kappa"], (float, str))


def test_all_functions_importable():
    """Test that all overall_funcs functions can be imported."""
    from pycm.overall_funcs import (
        log_loss_calc, brier_score_calc, alpha2_calc, alpha_calc,
        weighted_alpha_calc, B_calc, ARI_calc, pearson_C_calc,
        RCI_calc, AUNP_calc, CBA_calc, RR_calc, overall_MCC_calc,
        convex_combination, overall_CEN_calc, ncr, p_value_calc,
        NIR_calc, hamming_calc, zero_one_loss_calc, entropy_calc,
        weighted_kappa_calc, kappa_no_prevalence_calc, cross_entropy_calc,
        joint_entropy_calc, conditional_entropy_calc, mutual_information_calc,
        kl_divergence_calc, lambda_B_calc, lambda_A_calc, chi_square_calc,
        phi_square_calc, cramers_V_calc, DF_calc, reliability_calc,
        micro_calc, macro_calc, PC_AC1_calc, PC_S_calc,
        overall_jaccard_index_calc, overall_accuracy_calc,
        overall_random_accuracy_calc, overall_statistics
    )

    # All functions should be callable
    assert callable(overall_statistics)
    assert callable(reliability_calc)
    assert callable(ncr)
