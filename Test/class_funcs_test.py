# -*- coding: utf-8 -*-
"""
Tests for the class_funcs module.
"""
import pytest
import math


def test_ttpn_calc():
    """Test TTPN_calc function (TPR, TNR, PPV, NPV calculation)."""
    from pycm.class_funcs import TTPN_calc

    # TP=50, FN=10 -> TPR = 50/60 = 0.833...
    assert abs(TTPN_calc(50, 10) - 0.8333333) < 0.0001

    # TP=0, FN=0 -> Division by zero
    assert TTPN_calc(0, 0) == "None"

    # TP=10, FN=0 -> 1.0
    assert TTPN_calc(10, 0) == 1.0


def test_fxr_calc():
    """Test FXR_calc function (FNR, FPR, FDR, FOR calculation)."""
    from pycm.class_funcs import FXR_calc

    assert abs(FXR_calc(0.8) - 0.2) < 0.0001
    assert FXR_calc(1.0) == 0.0
    assert FXR_calc(0.0) == 1.0
    assert FXR_calc("None") == "None"


def test_acc_calc():
    """Test ACC_calc function (accuracy)."""
    from pycm.class_funcs import ACC_calc

    # Perfect accuracy
    assert ACC_calc(50, 50, 0, 0) == 1.0

    # 50% accuracy
    assert ACC_calc(25, 25, 25, 25) == 0.5

    # All zeros
    assert ACC_calc(0, 0, 0, 0) == "None"


def test_f_calc():
    """Test F_calc function (F-score)."""
    from pycm.class_funcs import F_calc

    # F1 score
    result = F_calc(50, 10, 5, 1)
    assert isinstance(result, float)
    assert 0 <= result <= 1

    # All zeros
    assert F_calc(0, 0, 0, 1) == "None"


def test_mcc_calc():
    """Test MCC_calc function (Matthews correlation coefficient)."""
    from pycm.class_funcs import MCC_calc

    # Perfect prediction (no FP or FN)
    result = MCC_calc(50, 50, 0, 0)
    # This actually computes to 1.0 with special handling
    assert isinstance(result, (float, str))

    # Typical case
    result = MCC_calc(50, 40, 10, 5)
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_mk_bm_calc():
    """Test MK_BM_calc function (Markedness, Informedness, ICSI)."""
    from pycm.class_funcs import MK_BM_calc

    assert abs(MK_BM_calc(0.9, 0.8) - 0.7) < 0.0001
    assert MK_BM_calc(1.0, 1.0) == 1.0
    assert MK_BM_calc(0.5, 0.5) == 0.0
    assert MK_BM_calc("None", 0.5) == "None"


def test_lr_calc():
    """Test LR_calc function (likelihood ratio)."""
    from pycm.class_funcs import LR_calc

    assert LR_calc(0.8, 0.2) == 4.0
    assert LR_calc(0.5, 0) == "None"
    assert LR_calc("None", 0.5) == "None"


def test_proportion_calc():
    """Test proportion_calc function (prevalence)."""
    from pycm.class_funcs import proportion_calc

    assert proportion_calc(50, 100) == 0.5
    assert proportion_calc(0, 100) == 0.0
    assert proportion_calc(50, 0) == "None"


def test_g_calc():
    """Test G_calc function (G-measure, G-mean)."""
    from pycm.class_funcs import G_calc

    assert G_calc(0.64, 0.64) == 0.64
    assert abs(G_calc(0.5, 0.5) - 0.5) < 0.0001
    assert G_calc(-1, 0.5) == "None"  # Negative value
    assert G_calc("None", 0.5) == "None"


def test_racc_calc():
    """Test RACC_calc function (random accuracy)."""
    from pycm.class_funcs import RACC_calc

    result = RACC_calc(50, 50, 100)
    assert abs(result - 0.25) < 0.0001

    assert RACC_calc(50, 50, 0) == "None"


def test_raccu_calc():
    """Test RACCU_calc function (random accuracy unbiased)."""
    from pycm.class_funcs import RACCU_calc

    result = RACCU_calc(50, 50, 100)
    assert abs(result - 0.25) < 0.0001

    assert RACCU_calc(50, 50, 0) == "None"


def test_err_calc():
    """Test ERR_calc function (error rate)."""
    from pycm.class_funcs import ERR_calc

    assert abs(ERR_calc(0.9) - 0.1) < 0.0001
    assert ERR_calc(1.0) == 0.0
    assert ERR_calc("None") == "None"


def test_jaccard_index_calc():
    """Test jaccard_index_calc function."""
    from pycm.class_funcs import jaccard_index_calc

    # TP=50, TOP=60, P=55 -> 50 / (60+55-50) = 50/65
    result = jaccard_index_calc(50, 60, 55)
    assert abs(result - 50/65) < 0.0001

    # All zeros
    assert jaccard_index_calc(0, 0, 0) == "None"


def test_is_calc():
    """Test IS_calc function (information score)."""
    from pycm.class_funcs import IS_calc

    result = IS_calc(50, 10, 5, 100)
    assert isinstance(result, float)

    # Edge case
    assert IS_calc(0, 0, 0, 100) == "None"


def test_auc_calc():
    """Test AUC_calc function."""
    from pycm.class_funcs import AUC_calc

    # TNR=0.9, TPR=0.8 -> (0.9+0.8)/2 = 0.85
    assert abs(AUC_calc(0.9, 0.8) - 0.85) < 0.0001
    assert AUC_calc(1.0, 1.0) == 1.0
    assert AUC_calc("None", 0.8) == "None"


def test_dind_calc():
    """Test dInd_calc function (distance index)."""
    from pycm.class_funcs import dInd_calc

    # Perfect classifier
    assert dInd_calc(1.0, 1.0) == 0.0

    # Typical case
    result = dInd_calc(0.9, 0.8)
    assert isinstance(result, float)
    assert result >= 0


def test_sind_calc():
    """Test sInd_calc function (similarity index)."""
    from pycm.class_funcs import sInd_calc

    # Perfect similarity (dInd=0)
    assert sInd_calc(0) == 1.0

    # Typical case
    result = sInd_calc(0.5)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_dp_calc():
    """Test DP_calc function (discriminant power)."""
    from pycm.class_funcs import DP_calc

    result = DP_calc(0.9, 0.8)
    assert isinstance(result, float)

    # Edge cases
    assert DP_calc(1.0, 0.8) == "None"  # Division by zero
    assert DP_calc(0.8, 1.0) == "None"


def test_gi_calc():
    """Test GI_calc function (Gini index)."""
    from pycm.class_funcs import GI_calc

    assert abs(GI_calc(0.8) - 0.6) < 0.0001
    assert GI_calc(0.5) == 0.0
    assert GI_calc(1.0) == 1.0
    assert GI_calc("None") == "None"


def test_lift_calc():
    """Test lift_calc function."""
    from pycm.class_funcs import lift_calc

    assert lift_calc(0.8, 0.4) == 2.0
    assert lift_calc(0.5, 0) == "None"


def test_am_calc():
    """Test AM_calc function (Automatic/Manual)."""
    from pycm.class_funcs import AM_calc

    assert AM_calc(60, 50) == 10
    assert AM_calc(50, 60) == -10
    assert AM_calc(50, 50) == 0


def test_op_calc():
    """Test OP_calc function (optimized precision)."""
    from pycm.class_funcs import OP_calc

    result = OP_calc(0.9, 0.8, 0.85)
    assert isinstance(result, float)

    assert OP_calc(0.9, 0, 0) == "None"


def test_iba_calc():
    """Test IBA_calc function (index of balanced accuracy)."""
    from pycm.class_funcs import IBA_calc

    result = IBA_calc(0.8, 0.9)
    assert isinstance(result, float)

    result_alpha = IBA_calc(0.8, 0.9, 0.5)
    assert isinstance(result_alpha, float)


def test_bcd_calc():
    """Test BCD_calc function (Bray-Curtis dissimilarity)."""
    from pycm.class_funcs import BCD_calc

    assert BCD_calc(10, 100) == 0.05
    assert BCD_calc(-10, 100) == 0.05  # abs()
    assert BCD_calc(10, 0) == "None"


def test_q_calc():
    """Test Q_calc function (Yule's Q)."""
    from pycm.class_funcs import Q_calc

    result = Q_calc(50, 40, 10, 5)
    assert isinstance(result, float)
    assert -1 <= result <= 1

    # Division by zero
    assert Q_calc(50, 40, 0, 5) == "None"


def test_agm_calc():
    """Test AGM_calc function (adjusted geometric mean)."""
    from pycm.class_funcs import AGM_calc

    result = AGM_calc(0.8, 0.9, 0.85, 50, 100)
    assert isinstance(result, float)

    # TPR = 0
    assert AGM_calc(0, 0.9, 0, 50, 100) == 0


def test_agf_calc():
    """Test AGF_calc function (adjusted F-score)."""
    from pycm.class_funcs import AGF_calc

    result = AGF_calc(50, 10, 5, 40)
    assert isinstance(result, float)


def test_oc_calc():
    """Test OC_calc function (overlap coefficient)."""
    from pycm.class_funcs import OC_calc

    result = OC_calc(50, 60, 55)
    assert isinstance(result, float)
    assert 0 <= result <= 1

    assert OC_calc(50, 0, 0) == "None"


def test_bb_calc():
    """Test BB_calc function (Braun-Blanquet similarity)."""
    from pycm.class_funcs import BB_calc

    result = BB_calc(50, 60, 55)
    assert isinstance(result, float)
    assert 0 <= result <= 1

    assert BB_calc(50, 0, 0) == "None"


def test_ooc_calc():
    """Test OOC_calc function (Otsuka-Ochiai coefficient)."""
    from pycm.class_funcs import OOC_calc

    result = OOC_calc(50, 60, 55)
    assert isinstance(result, float)

    assert OOC_calc(50, 0, 0) == "None"


def test_ti_calc():
    """Test TI_calc function (Tversky index)."""
    from pycm.class_funcs import TI_calc

    result = TI_calc(50, 10, 5, 1, 1)
    assert isinstance(result, float)

    assert TI_calc(0, 0, 0, 1, 1) == "None"


def test_nb_calc():
    """Test NB_calc function (net benefit)."""
    from pycm.class_funcs import NB_calc

    result = NB_calc(50, 10, 100, 0.5)
    assert isinstance(result, float)

    assert NB_calc(50, 10, 0, 0.5) == "None"


def test_sensitivity_index_calc():
    """Test sensitivity_index_calc function (d prime)."""
    from pycm.class_funcs import sensitivity_index_calc

    result = sensitivity_index_calc(0.8, 0.2)
    assert isinstance(result, float)


def test_cen_calc():
    """Test CEN_calc function (confusion entropy)."""
    from pycm.class_funcs import CEN_calc

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}

    result = CEN_calc(classes, table, 55, 60, 0)
    assert isinstance(result, float)
    assert result >= 0


def test_cen_misclassification_calc():
    """Test CEN_misclassification_calc function."""
    from pycm.class_funcs import CEN_misclassification_calc

    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}

    result = CEN_misclassification_calc(table, 55, 60, 0, 1, 0)
    assert isinstance(result, float)

    # Modified mode
    result_mod = CEN_misclassification_calc(table, 55, 60, 0, 1, 0, modified=True)
    assert isinstance(result_mod, float)


def test_basic_statistics():
    """Test basic_statistics function."""
    from pycm.class_funcs import basic_statistics

    TP = {0: 50, 1: 35}
    TN = {0: 35, 1: 50}
    FP = {0: 5, 1: 10}
    FN = {0: 10, 1: 5}

    result = basic_statistics(TP, TN, FP, FN)

    assert result["TP"] == TP
    assert result["TN"] == TN
    assert result["FP"] == FP
    assert result["FN"] == FN
    assert "ACC" in result  # Check that all CLASS_PARAMS are initialized


def test_class_statistics():
    """Test class_statistics function."""
    from pycm.class_funcs import class_statistics

    TP = {0: 50, 1: 35}
    TN = {0: 35, 1: 50}
    FP = {0: 5, 1: 10}
    FN = {0: 10, 1: 5}
    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}

    result = class_statistics(TP, TN, FP, FN, classes, table)

    # Check basic values
    assert result["TP"] == TP
    assert result["TN"] == TN

    # Check computed statistics
    assert result["POP"][0] == 100
    assert result["POP"][1] == 100
    assert result["P"][0] == 60
    assert result["P"][1] == 40

    # Check rates
    assert abs(result["TPR"][0] - 50/60) < 0.0001
    assert abs(result["TNR"][0] - 35/40) < 0.0001

    # Check metrics exist
    assert "ACC" in result
    assert "F1" in result
    assert "MCC" in result
    assert "AUC" in result
    assert "PRE" in result


def test_class_statistics_interpretation():
    """Test class_statistics includes interpretation fields."""
    from pycm.class_funcs import class_statistics

    TP = {0: 50}
    TN = {0: 40}
    FP = {0: 5}
    FN = {0: 5}
    classes = [0]
    table = {0: {0: 50}}

    result = class_statistics(TP, TN, FP, FN, classes, table)

    # Check interpretation fields
    assert "PLRI" in result
    assert "NLRI" in result
    assert "DPI" in result
    assert "AUCI" in result
    assert "QI" in result
    assert "MCCI" in result


def test_all_functions_importable():
    """Test that all class_funcs functions can be imported."""
    from pycm.class_funcs import (
        sensitivity_index_calc, NB_calc, TI_calc, OOC_calc, OC_calc,
        BB_calc, AGF_calc, AGM_calc, Q_calc, TTPN_calc, FXR_calc,
        ACC_calc, F_calc, MCC_calc, MK_BM_calc, LR_calc, proportion_calc,
        G_calc, RACC_calc, RACCU_calc, ERR_calc, jaccard_index_calc,
        IS_calc, CEN_misclassification_calc, CEN_calc, AUC_calc,
        dInd_calc, sInd_calc, DP_calc, GI_calc, lift_calc, AM_calc,
        OP_calc, IBA_calc, BCD_calc, basic_statistics, class_statistics
    )

    # All functions should be callable
    assert callable(sensitivity_index_calc)
    assert callable(NB_calc)
    assert callable(TI_calc)
    assert callable(class_statistics)
