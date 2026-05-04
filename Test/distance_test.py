# -*- coding: utf-8 -*-
"""Tests for the distance module."""
import pytest
import math


def test_distance_type_enum():
    """Test DistanceType enum values."""
    from pycm.distance import DistanceType

    assert DistanceType.AMPLE.value == "AMPLE"
    assert DistanceType.Anderberg.value == "Anderberg"
    assert DistanceType.AndresMarzoDelta.value == "AndresMarzoDelta"
    assert DistanceType.BaroniUrbaniBuserI.value == "BaroniUrbaniBuserI"
    assert DistanceType.BaulieuI.value == "BaulieuI"
    assert DistanceType.Hamann.value == "Hamann"
    assert DistanceType.KuhnsVII.value == "KuhnsVII"


def test_distance_type_enum_count():
    """Test that DistanceType enum has expected number of members."""
    from pycm.distance import DistanceType

    # 63 distance types in the enum
    assert len(DistanceType) == 63


def test_ample_calc():
    """Test AMPLE_calc function."""
    from pycm.distance import AMPLE_calc

    # TP=50, FP=10, FN=5, TN=35
    # part1 = 50/(50+10) = 0.833...
    # part2 = 5/(5+35) = 0.125
    # result = |0.833 - 0.125| = 0.708...
    result = AMPLE_calc(50, 10, 5, 35)
    assert isinstance(result, float)
    assert abs(result - 0.7083333) < 0.001

    # Edge case: division by zero
    assert AMPLE_calc(0, 0, 0, 0) == "None"


def test_anderberg_calc():
    """Test Anderberg_calc function."""
    from pycm.distance import Anderberg_calc

    result = Anderberg_calc(50, 10, 5, 35)
    assert isinstance(result, float)

    # All zeros
    assert Anderberg_calc(0, 0, 0, 0) == "None"


def test_andres_marzo_delta_calc():
    """Test AndresMarzoDelta_calc function."""
    from pycm.distance import AndresMarzoDelta_calc

    result = AndresMarzoDelta_calc(50, 10, 5, 35)
    assert isinstance(result, float)

    # With FP=FN=0, sqrt(0) = 0
    result_zero = AndresMarzoDelta_calc(50, 0, 0, 50)
    assert abs(result_zero - 1.0) < 0.001  # (50+50-0)/100 = 1.0


def test_baroni_urbani_buser_calc():
    """Test BaroniUrbaniBuserI_calc and II functions."""
    from pycm.distance import BaroniUrbaniBuserI_calc, BaroniUrbaniBuserII_calc

    result_I = BaroniUrbaniBuserI_calc(50, 10, 5, 35)
    assert isinstance(result_I, float)
    assert 0 <= result_I <= 1

    result_II = BaroniUrbaniBuserII_calc(50, 10, 5, 35)
    assert isinstance(result_II, float)
    assert -1 <= result_II <= 1


def test_batagelj_bren_calc():
    """Test BatageljBren_calc function."""
    from pycm.distance import BatageljBren_calc

    # (FP * FN) / (TP * TN) = (10 * 5) / (50 * 35) = 50/1750
    result = BatageljBren_calc(50, 10, 5, 35)
    assert abs(result - 50/1750) < 0.0001

    # Division by zero when TP*TN = 0
    assert BatageljBren_calc(0, 10, 5, 35) == "None"


def test_baulieu_variants_calc():
    """Test various Baulieu distance functions."""
    from pycm.distance import (
        BaulieuI_calc, BaulieuII_calc, BaulieuIII_calc, BaulieuIV_calc,
        BaulieuV_calc, BaulieuVI_calc, BaulieuVII_calc, BaulieuVIII_calc,
        BaulieuIX_calc, BaulieuX_calc, BaulieuXI_calc, BaulieuXII_calc,
        BaulieuXIII_calc, BaulieuXIV_calc, BaulieuXV_calc
    )

    TP, FP, FN, TN = 50, 10, 5, 35

    # Test all Baulieu variants return valid results
    assert isinstance(BaulieuI_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuII_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuIII_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuIV_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuV_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuVI_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuVII_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuVIII_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuIX_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuX_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuXI_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuXII_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuXIII_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuXIV_calc(TP, FP, FN, TN), float)
    assert isinstance(BaulieuXV_calc(TP, FP, FN, TN), float)


def test_baulieu_v_specific():
    """Test BaulieuV_calc specific calculation."""
    from pycm.distance import BaulieuV_calc

    # (FP + FN + 1) / (TP + FP + FN + 1) = (10 + 5 + 1) / (50 + 10 + 5 + 1) = 16/66
    result = BaulieuV_calc(50, 10, 5, 35)
    assert abs(result - 16/66) < 0.0001


def test_benini_calc():
    """Test BeniniI_calc and BeniniII_calc functions."""
    from pycm.distance import BeniniI_calc, BeniniII_calc

    result_I = BeniniI_calc(50, 10, 5, 35)
    assert isinstance(result_I, float)

    result_II = BeniniII_calc(50, 10, 5, 35)
    assert isinstance(result_II, float)


def test_canberra_calc():
    """Test Canberra_calc function."""
    from pycm.distance import Canberra_calc

    # (FP + FN) / ((TP + FP) + (TP + FN)) = 15 / (60 + 55) = 15/115
    result = Canberra_calc(50, 10, 5, 35)
    assert abs(result - 15/115) < 0.0001

    # Edge case
    assert Canberra_calc(0, 0, 0, 0) == "None"


def test_clement_calc():
    """Test Clement_calc function."""
    from pycm.distance import Clement_calc

    result = Clement_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_consonni_todeschini_calc():
    """Test Consonni & Todeschini functions."""
    from pycm.distance import (
        ConsonniTodeschiniI_calc, ConsonniTodeschiniII_calc,
        ConsonniTodeschiniIII_calc, ConsonniTodeschiniIV_calc,
        ConsonniTodeschiniV_calc
    )

    TP, FP, FN, TN = 50, 10, 5, 35

    result_I = ConsonniTodeschiniI_calc(TP, FP, FN, TN)
    assert isinstance(result_I, float)
    assert 0 <= result_I <= 1

    result_II = ConsonniTodeschiniII_calc(TP, FP, FN, TN)
    assert isinstance(result_II, float)

    result_III = ConsonniTodeschiniIII_calc(TP, FP, FN, TN)
    assert isinstance(result_III, float)

    result_IV = ConsonniTodeschiniIV_calc(TP, FP, FN, TN)
    assert isinstance(result_IV, float)

    result_V = ConsonniTodeschiniV_calc(TP, FP, FN, TN)
    assert isinstance(result_V, float)


def test_dennis_calc():
    """Test Dennis_calc function."""
    from pycm.distance import Dennis_calc

    result = Dennis_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_digby_calc():
    """Test Digby_calc function."""
    from pycm.distance import Digby_calc

    result = Digby_calc(50, 10, 5, 35)
    assert isinstance(result, float)
    assert -1 <= result <= 1

    # Edge case: one of FP*FN = 0 should still work
    result_zero = Digby_calc(50, 0, 0, 50)
    assert isinstance(result_zero, float)


def test_dispersion_calc():
    """Test Dispersion_calc function."""
    from pycm.distance import Dispersion_calc

    # (TP*TN - FP*FN) / n^2 = (50*35 - 10*5) / 100^2 = 1700/10000 = 0.17
    result = Dispersion_calc(50, 10, 5, 35)
    assert abs(result - 0.17) < 0.001


def test_doolittle_calc():
    """Test Doolittle_calc function."""
    from pycm.distance import Doolittle_calc

    result = Doolittle_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_eyraud_calc():
    """Test Eyraud_calc function."""
    from pycm.distance import Eyraud_calc

    result = Eyraud_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_fager_mcgowan_calc():
    """Test FagerMcGowan_calc function."""
    from pycm.distance import FagerMcGowan_calc

    result = FagerMcGowan_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_faith_calc():
    """Test Faith_calc function."""
    from pycm.distance import Faith_calc

    # (TP + TN/2) / n = (50 + 17.5) / 100 = 0.675
    result = Faith_calc(50, 10, 5, 35)
    assert abs(result - 0.675) < 0.001


def test_fleiss_levin_paik_calc():
    """Test FleissLevinPaik_calc function."""
    from pycm.distance import FleissLevinPaik_calc

    # 2*TN / (2*TN + FP + FN) = 70 / (70 + 15) = 70/85
    result = FleissLevinPaik_calc(50, 10, 5, 35)
    assert abs(result - 70/85) < 0.001


def test_forbes_calc():
    """Test ForbesI_calc and ForbesII_calc functions."""
    from pycm.distance import ForbesI_calc, ForbesII_calc

    result_I = ForbesI_calc(50, 10, 5, 35)
    assert isinstance(result_I, float)

    result_II = ForbesII_calc(50, 10, 5, 35)
    assert isinstance(result_II, float)


def test_fossum_calc():
    """Test Fossum_calc function."""
    from pycm.distance import Fossum_calc

    result = Fossum_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_gilbert_wells_calc():
    """Test GilbertWells_calc function."""
    from pycm.distance import GilbertWells_calc

    # This uses factorials, so small values work best
    result = GilbertWells_calc(5, 2, 1, 4)
    assert isinstance(result, float)

    # Large factorials may overflow
    result_large = GilbertWells_calc(50, 10, 5, 35)
    # May return "None" due to overflow or valid float
    assert isinstance(result_large, (float, str))


def test_goodall_calc():
    """Test Goodall_calc function."""
    from pycm.distance import Goodall_calc

    result = Goodall_calc(50, 10, 5, 35)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_goodman_kruskal_lambda_calc():
    """Test GoodmanKruskalLambda_calc and GoodmanKruskalLambdaR_calc functions."""
    from pycm.distance import GoodmanKruskalLambda_calc, GoodmanKruskalLambdaR_calc

    result = GoodmanKruskalLambda_calc(50, 10, 5, 35)
    assert isinstance(result, float)

    result_r = GoodmanKruskalLambdaR_calc(50, 10, 5, 35)
    assert isinstance(result_r, float)


def test_guttman_lambda_calc():
    """Test GuttmanLambdaA_calc and GuttmanLambdaB_calc functions."""
    from pycm.distance import GuttmanLambdaA_calc, GuttmanLambdaB_calc

    result_A = GuttmanLambdaA_calc(50, 10, 5, 35)
    assert isinstance(result_A, float)

    result_B = GuttmanLambdaB_calc(50, 10, 5, 35)
    assert isinstance(result_B, float)


def test_hamann_calc():
    """Test Hamann_calc function."""
    from pycm.distance import Hamann_calc

    # (TP + TN - FP - FN) / n = (50 + 35 - 10 - 5) / 100 = 70/100 = 0.7
    result = Hamann_calc(50, 10, 5, 35)
    assert abs(result - 0.7) < 0.001


def test_harris_lahey_calc():
    """Test HarrisLahey_calc function."""
    from pycm.distance import HarrisLahey_calc

    result = HarrisLahey_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_hawkins_dotson_calc():
    """Test HawkinsDotson_calc function."""
    from pycm.distance import HawkinsDotson_calc

    result = HawkinsDotson_calc(50, 10, 5, 35)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_kendall_tau_calc():
    """Test KendallTau_calc function."""
    from pycm.distance import KendallTau_calc

    result = KendallTau_calc(50, 10, 5, 35)
    assert isinstance(result, float)


def test_kent_foster_calc():
    """Test KentFosterI_calc and KentFosterII_calc functions."""
    from pycm.distance import KentFosterI_calc, KentFosterII_calc

    result_I = KentFosterI_calc(50, 10, 5, 35)
    assert isinstance(result_I, float)

    result_II = KentFosterII_calc(50, 10, 5, 35)
    assert isinstance(result_II, float)


def test_koppen_calc():
    """Test KoppenI_calc and KoppenII_calc functions."""
    from pycm.distance import KoppenI_calc, KoppenII_calc

    result_I = KoppenI_calc(50, 10, 5, 35)
    assert isinstance(result_I, float)

    # KoppenII = TP + (FP + FN) / 2 = 50 + 7.5 = 57.5
    result_II = KoppenII_calc(50, 10, 5, 35)
    assert abs(result_II - 57.5) < 0.001


def test_kuder_richardson_calc():
    """Test KuderRichardson_calc function."""
    from pycm.distance import KuderRichardson_calc

    result = KuderRichardson_calc(50, 10, 5, 35)
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_kuhns_variants_calc():
    """Test various Kuhns distance functions."""
    from pycm.distance import (
        KuhnsI_calc, KuhnsII_calc, KuhnsIII_calc, KuhnsIV_calc,
        KuhnsV_calc, KuhnsVI_calc, KuhnsVII_calc
    )

    TP, FP, FN, TN = 50, 10, 5, 35

    result_I = KuhnsI_calc(TP, FP, FN, TN)
    assert isinstance(result_I, float)

    result_II = KuhnsII_calc(TP, FP, FN, TN)
    assert isinstance(result_II, float)

    result_III = KuhnsIII_calc(TP, FP, FN, TN)
    assert isinstance(result_III, float)

    result_IV = KuhnsIV_calc(TP, FP, FN, TN)
    assert isinstance(result_IV, float)

    result_V = KuhnsV_calc(TP, FP, FN, TN)
    assert isinstance(result_V, float)

    result_VI = KuhnsVI_calc(TP, FP, FN, TN)
    assert isinstance(result_VI, float)

    result_VII = KuhnsVII_calc(TP, FP, FN, TN)
    assert isinstance(result_VII, float)


def test_distance_mapper():
    """Test DISTANCE_MAPPER dictionary."""
    from pycm.distance import DISTANCE_MAPPER, DistanceType

    # Check that all distance types are mapped
    assert len(DISTANCE_MAPPER) == len(DistanceType)

    # Check that all enum values have a corresponding function
    for distance_type in DistanceType:
        assert distance_type in DISTANCE_MAPPER
        assert callable(DISTANCE_MAPPER[distance_type])


def test_distance_mapper_invocation():
    """Test calling functions through DISTANCE_MAPPER."""
    from pycm.distance import DISTANCE_MAPPER, DistanceType

    TP, FP, FN, TN = 50, 10, 5, 35

    # Test a few functions through the mapper
    ample_result = DISTANCE_MAPPER[DistanceType.AMPLE](TP, FP, FN, TN)
    assert isinstance(ample_result, float)

    hamann_result = DISTANCE_MAPPER[DistanceType.Hamann](TP, FP, FN, TN)
    assert abs(hamann_result - 0.7) < 0.001

    faith_result = DISTANCE_MAPPER[DistanceType.Faith](TP, FP, FN, TN)
    assert abs(faith_result - 0.675) < 0.001


def test_edge_cases_all_zeros():
    """Test functions with all zero inputs."""
    from pycm.distance import (
        AMPLE_calc, Anderberg_calc, Canberra_calc, Faith_calc, Hamann_calc
    )

    # Most functions should return "None" with all zeros
    assert AMPLE_calc(0, 0, 0, 0) == "None"
    assert Anderberg_calc(0, 0, 0, 0) == "None"
    assert Canberra_calc(0, 0, 0, 0) == "None"
    assert Faith_calc(0, 0, 0, 0) == "None"
    assert Hamann_calc(0, 0, 0, 0) == "None"


def test_edge_cases_perfect_classifier():
    """Test functions with perfect classifier (FP=FN=0)."""
    from pycm.distance import (
        AMPLE_calc, Hamann_calc, Dispersion_calc, AndresMarzoDelta_calc
    )

    TP, FP, FN, TN = 50, 0, 0, 50

    # AMPLE = |50/50 - 0/50| = |1 - 0| = 1
    assert abs(AMPLE_calc(TP, FP, FN, TN) - 1.0) < 0.001

    # Hamann = (50 + 50 - 0 - 0) / 100 = 1.0
    assert abs(Hamann_calc(TP, FP, FN, TN) - 1.0) < 0.001

    # Dispersion = (50*50 - 0*0) / 100^2 = 2500/10000 = 0.25
    assert abs(Dispersion_calc(TP, FP, FN, TN) - 0.25) < 0.001

    # AndresMarzoDelta = (50 + 50 - 0) / 100 = 1.0
    assert abs(AndresMarzoDelta_calc(TP, FP, FN, TN) - 1.0) < 0.001


def test_all_functions_importable():
    """Test that all distance functions can be imported."""
    from pycm.distance import (
        DistanceType, DISTANCE_MAPPER,
        AMPLE_calc, Anderberg_calc, AndresMarzoDelta_calc,
        BaroniUrbaniBuserI_calc, BaroniUrbaniBuserII_calc,
        BatageljBren_calc,
        BaulieuI_calc, BaulieuII_calc, BaulieuIII_calc, BaulieuIV_calc,
        BaulieuV_calc, BaulieuVI_calc, BaulieuVII_calc, BaulieuVIII_calc,
        BaulieuIX_calc, BaulieuX_calc, BaulieuXI_calc, BaulieuXII_calc,
        BaulieuXIII_calc, BaulieuXIV_calc, BaulieuXV_calc,
        BeniniI_calc, BeniniII_calc,
        Canberra_calc, Clement_calc,
        ConsonniTodeschiniI_calc, ConsonniTodeschiniII_calc,
        ConsonniTodeschiniIII_calc, ConsonniTodeschiniIV_calc,
        ConsonniTodeschiniV_calc,
        Dennis_calc, Digby_calc, Dispersion_calc, Doolittle_calc, Eyraud_calc,
        FagerMcGowan_calc, Faith_calc, FleissLevinPaik_calc,
        ForbesI_calc, ForbesII_calc, Fossum_calc,
        GilbertWells_calc, Goodall_calc,
        GoodmanKruskalLambda_calc, GoodmanKruskalLambdaR_calc,
        GuttmanLambdaA_calc, GuttmanLambdaB_calc,
        Hamann_calc, HarrisLahey_calc, HawkinsDotson_calc,
        KendallTau_calc, KentFosterI_calc, KentFosterII_calc,
        KoppenI_calc, KoppenII_calc, KuderRichardson_calc,
        KuhnsI_calc, KuhnsII_calc, KuhnsIII_calc, KuhnsIV_calc,
        KuhnsV_calc, KuhnsVI_calc, KuhnsVII_calc
    )

    # All functions should be callable
    assert callable(AMPLE_calc)
    assert callable(Anderberg_calc)
    assert callable(Hamann_calc)
    assert callable(KuhnsVII_calc)
    assert len(DISTANCE_MAPPER) == 63
