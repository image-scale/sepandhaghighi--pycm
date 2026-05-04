# -*- coding: utf-8 -*-
"""
Tests for the output module.
"""
import pytest


def test_color_check():
    """Test color_check function."""
    from pycm.output import color_check

    # Tuple input
    assert color_check((255, 0, 0)) == [255, 0, 0]

    # List input
    assert color_check([0, 255, 0]) == [0, 255, 0]

    # Named color
    assert color_check("red") == [255, 0, 0]
    assert color_check("green") == [0, 128, 0]  # green is 0,128,0 in HTML colors
    assert color_check("blue") == [0, 0, 255]

    # Invalid input
    assert color_check("invalid") == [0, 0, 0]
    assert color_check(None) == [0, 0, 0]
    assert color_check([300, 0, 0]) == [0, 0, 0]  # Out of range


def test_html_table_color():
    """Test html_table_color function."""
    from pycm.output import html_table_color

    row = {0: 50, 1: 10}

    # Get color for a cell
    result = html_table_color(row, 50, (255, 0, 0))
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(0 <= c <= 255 for c in result)


def test_html_dataset_type():
    """Test html_dataset_type function."""
    from pycm.output import html_dataset_type

    # Binary balanced
    result = html_dataset_type(is_binary=True, is_imbalanced=False)
    assert "Binary Classification" in result
    assert "Balanced" in result

    # Multi-class imbalanced
    result = html_dataset_type(is_binary=False, is_imbalanced=True)
    assert "Multi-Class Classification" in result
    assert "Imbalanced" in result


def test_html_table():
    """Test html_table function."""
    from pycm.output import html_table

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}

    result = html_table(classes, table, (255, 0, 0))
    assert "<table" in result
    assert "Confusion Matrix" in result

    # Normalized
    result_norm = html_table(classes, table, (0, 255, 0), normalize=True)
    assert "Normalized" in result_norm


def test_html_overall_stat():
    """Test html_overall_stat function."""
    from pycm.output import html_overall_stat

    overall_stat = {"Kappa": 0.7, "Overall ACC": 0.85}

    result = html_overall_stat(overall_stat, digit=3)
    assert "<table" in result
    assert "Overall Statistics" in result


def test_html_overall_stat_empty():
    """Test html_overall_stat with empty input."""
    from pycm.output import html_overall_stat

    result = html_overall_stat({}, digit=3)
    assert result == ""


def test_html_class_stat():
    """Test html_class_stat function."""
    from pycm.output import html_class_stat

    classes = [0, 1]
    class_stat = {"TPR": {0: 0.833, 1: 0.875}, "TNR": {0: 0.875, 1: 0.833}}

    result = html_class_stat(classes, class_stat, digit=3)
    assert "<table" in result
    assert "Class Statistics" in result


def test_html_class_stat_empty():
    """Test html_class_stat with empty input."""
    from pycm.output import html_class_stat

    result = html_class_stat([], {}, digit=3)
    assert result == ""


def test_table_print():
    """Test table_print function."""
    from pycm.output import table_print

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}

    result = table_print(classes, table)
    assert "Predict" in result
    assert "Actual" in result
    assert "50" in result
    assert "35" in result


def test_sparse_table_print():
    """Test sparse_table_print function."""
    from pycm.output import sparse_table_print

    sparse_matrix = (
        {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}},
        [0, 1],
        [0, 1]
    )

    result = sparse_table_print(sparse_matrix)
    assert "Predict" in result
    assert "Actual" in result


def test_csv_matrix_print():
    """Test csv_matrix_print function."""
    from pycm.output import csv_matrix_print

    classes = [0, 1]
    table = {0: {0: 50, 1: 10}, 1: {0: 5, 1: 35}}

    # Without header
    result = csv_matrix_print(classes, table)
    assert "50,10" in result
    assert "5,35" in result

    # With header
    result_header = csv_matrix_print(classes, table, header=True)
    assert '"0"' in result_header


def test_csv_print():
    """Test csv_print function."""
    from pycm.output import csv_print

    classes = [0, 1]
    class_stat = {"TPR": {0: 0.833, 1: 0.875}, "TNR": {0: 0.875, 1: 0.833}}

    result = csv_print(classes, class_stat, digit=3)
    assert "Class" in result
    assert "TPR" in result
    assert "TNR" in result


def test_csv_print_empty():
    """Test csv_print with empty input."""
    from pycm.output import csv_print

    result = csv_print([], {}, digit=3)
    assert result == ""


def test_stat_print():
    """Test stat_print function."""
    from pycm.output import stat_print

    classes = [0, 1]
    class_stat = {"TPR": {0: 0.833, 1: 0.875}}
    overall_stat = {"Kappa": 0.7}

    result = stat_print(classes, class_stat, overall_stat, digit=3)
    assert "Overall Statistics" in result
    assert "Class Statistics" in result


def test_compare_report_print():
    """Test compare_report_print function."""
    from pycm.output import compare_report_print

    sorted_list = ["cm1", "cm2"]
    scores = {
        "cm1": {"class": 0.9, "overall": 0.85},
        "cm2": {"class": 0.8, "overall": 0.75}
    }

    result = compare_report_print(sorted_list, scores, "cm1")
    assert "Best : cm1" in result
    assert "Rank" in result
    assert "Name" in result


def test_pycm_help(capsys):
    """Test pycm_help function."""
    from pycm.output import pycm_help

    pycm_help()
    captured = capsys.readouterr()
    assert "pycm" in captured.out.lower() or "github" in captured.out.lower()


def test_online_help_no_param(capsys):
    """Test online_help with no parameter."""
    from pycm.output import online_help

    # When no param is given, it prints available params
    online_help()
    captured = capsys.readouterr()
    assert "choose" in captured.out.lower() or "parameter" in captured.out.lower()


def test_all_functions_importable():
    """Test that all output functions can be imported."""
    from pycm.output import (
        html_dataset_type, color_check, html_table_color, html_table,
        html_overall_stat, html_class_stat, pycm_help, table_print,
        sparse_table_print, csv_matrix_print, csv_print, stat_print,
        compare_report_print, online_help
    )

    # All functions should be callable
    assert callable(html_dataset_type)
    assert callable(color_check)
    assert callable(html_table_color)
    assert callable(html_table)
    assert callable(table_print)
    assert callable(csv_print)
    assert callable(online_help)
