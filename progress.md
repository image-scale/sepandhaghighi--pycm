# Progress

(Updated after each feature commit.)

## Round 1
**Task**: Task 1 — Implement params module with version, constants, error messages, and parameter dictionaries
**Files created**: pycm/params.py, Test/params_test.py
**Commit**: Implement params module with version constants, error messages, parameter dictionaries, benchmark score mappings, HTML templates, color definitions, and mathematical coefficients
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 2
**Task**: Task 2 — Implement errors module with custom exception classes
**Files created**: pycm/errors.py, Test/error_classes_test.py
**Commit**: Implement errors module with pycmVectorError, pycmMatrixError, pycmCIError, pycmAverageError, pycmCompareError, pycmPlotError, pycmCurveError, pycmMultiLabelError
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 3
**Task**: Task 3 — Implement interpret module with analysis functions
**Files created**: pycm/interpret.py, Test/interpret_test.py
**Commit**: Implement interpret module with benchmark analysis functions (Q, MCC, NLR, PLR, DP, AUC, kappa benchmarks, lambda, alpha, pearson_C, V analysis)
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 4
**Task**: Task 4 — Implement utils module with validation and calculation helpers
**Files created**: pycm/utils.py, Test/utils_test.py
**Commit**: Implement utils module with vector/matrix validation, filtering, calculation helpers, normalization, sparse matrix, transpose, and mathematical utilities
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 5
**Task**: Task 5 — Implement ci module with confidence interval calculations
**Files created**: pycm/ci.py, Test/ci_test.py
**Commit**: Implement ci module with confidence interval calculations (normal approximation, Wilson, Agresti-Coull methods, SE calculations for AUC, LR, kappa)
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 6
**Task**: Task 6 — Implement class_funcs module with class-level statistics calculations
**Files created**: pycm/class_funcs.py, Test/class_funcs_test.py
**Commit**: Implement class_funcs module with 40+ class-level metrics (TPR, TNR, PPV, NPV, F-scores, MCC, AUC, CEN, Jaccard, Yule's Q, Gini index, etc.)
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 7
**Task**: Task 7 — Implement overall_funcs module with overall statistics calculations
**Files created**: pycm/overall_funcs.py, Test/overall_funcs_test.py
**Commit**: Implement overall_funcs module with 50+ overall metrics (Kappa, entropy, chi-squared, lambda, Bangdiwala's B, ARI, p-value, macro/micro averages)
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state
