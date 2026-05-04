# Todo

## Plan
Implement pycm in dependency order: start with foundational modules (params, errors), then utility functions (utils, interpret), then statistics calculation functions (class_funcs, overall_funcs, ci), followed by output formatting (output), handlers and core ConfusionMatrix class, and finally advanced features (Compare, curves, multilabel). Each feature commit includes both production code and corresponding tests.

## Tasks
- [x] Task 1: Implement params module with version, constants, error messages, and parameter dictionaries (pycm/params.py + Test/params_test.py)
- [x] Task 2: Implement errors module with custom exception classes for vector, matrix, CI, average, compare, plot, curve, and multilabel errors (pycm/errors.py + Test/error_classes_test.py)
- [ ] Task 3: Implement interpret module with analysis functions for kappa benchmarks, AUC, PLR, NLR, DP, MCC, Q, V, lambda, alpha, and Pearson C interpretation (pycm/interpret.py + Test/interpret_test.py)
- [ ] Task 4: Implement utils module with vector/matrix validation, filtering, calculation helpers, normalization, sparse matrix, transpose, and mathematical utilities (pycm/utils.py + Test/utils_test.py)
- [ ] Task 5: Implement ci module with confidence interval calculations for class and overall statistics including normal approximation, Wilson, and Agresti methods (pycm/ci.py + Test/ci_test.py)
- [ ] Task 6: Implement class_funcs module with class-level statistics calculations including TPR, TNR, PPV, NPV, F-scores, MCC, AUC, CEN, and 50+ other metrics (pycm/class_funcs.py + Test/class_funcs_test.py)
- [ ] Task 7: Implement overall_funcs module with overall statistics calculations including Kappa, accuracy, entropy measures, lambda, alpha, Brier score, and macro/micro averages (pycm/overall_funcs.py + Test/overall_funcs_test.py)
- [ ] Task 8: Implement output module with formatting functions for tables, statistics, HTML, CSV output, and online help functionality (pycm/output.py + Test/output_test.py)
- [ ] Task 9: Implement distance module with DistanceType enum and distance/similarity metric calculations (pycm/distance.py + Test/distance_test.py)
- [ ] Task 10: Implement handlers module with initialization handlers for ConfusionMatrix including vector, matrix, array, and file handlers (pycm/handlers.py + Test/handlers_test.py)
- [ ] Task 11: Implement core ConfusionMatrix class with initialization from vectors/matrix, stat methods, save/export methods, relabel, combine, position, and to_array (pycm/cm.py + pycm/__init__.py + Test/function_test.py)
- [ ] Task 12: Implement Compare class for comparing multiple confusion matrices with scoring and ranking functionality (pycm/compare.py + Test/compare_test.py)
- [ ] Task 13: Implement curve module with Curve base class and ROCCurve, PRCurve, PCurve, RCurve, F1Curve for threshold-based analysis (pycm/curve.py + Test/curve_test.py)
- [ ] Task 14: Implement MultiLabelCM class for multilabel confusion matrix support with class-wise and sample-wise modes (pycm/multilabel_cm.py + Test/multilabel_test.py)
- [ ] Task 15: Implement generate_random_data module with confusion matrix generation functions and benchmark utilities, plus __main__.py entry point (pycm/generate_random_data.py + pycm/__main__.py + Test/generate_data_test.py)
