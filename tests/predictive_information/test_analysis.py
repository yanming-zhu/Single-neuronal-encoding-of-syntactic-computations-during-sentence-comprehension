"""Behavioral tests for the unit/population predictive-information analysis."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_predictive_information_analysis_module_is_available() -> None:
    """Catch a missing reproducible analysis implementation."""
    try:
        module = importlib.import_module(
            "scripts.analyze_unit_population_predictive_information"
        )
    except ModuleNotFoundError:
        module = None

    assert module is not None


def analysis_module():
    return importlib.import_module(
        "scripts.analyze_unit_population_predictive_information"
    )


def test_predictive_information_is_log_likelihood_gain_in_bits() -> None:
    """Catch using AUC, accuracy, or natural logarithms as the information metric."""
    module = analysis_module()
    value = module.predictive_information_bits(
        y_true=np.array([1, 0]),
        model_probability=np.array([0.75, 0.25]),
        null_probability=np.array([0.50, 0.50]),
    )

    assert value == pytest.approx(0.5849625007211562)


def test_predictive_information_can_be_negative_for_harmful_predictions() -> None:
    """Catch clipping the metric at zero and hiding out-of-sample harm."""
    module = analysis_module()
    value = module.predictive_information_bits(
        y_true=np.array([1, 0]),
        model_probability=np.array([0.10, 0.90]),
        null_probability=np.array([0.50, 0.50]),
    )

    assert value == pytest.approx(-2.321928094887362)


def test_repeated_oof_predictions_are_averaged_once_per_trial() -> None:
    """Catch scoring repeated test appearances as independent observations."""
    module = analysis_module()
    probability, counts = module.aggregate_oof_probabilities(
        n_trials=3,
        test_indices=[np.array([0, 1]), np.array([0, 2])],
        probability_batches=[np.array([0.20, 0.80]), np.array([0.60, 0.30])],
    )

    np.testing.assert_allclose(probability, np.array([0.40, 0.80, 0.30]))
    np.testing.assert_array_equal(counts, np.array([2, 1, 1]))


def test_leave_one_sentence_pair_out_has_no_pair_leakage() -> None:
    """Catch placing trials from one sentence pair in both train and test."""
    module = analysis_module()
    pair_ids = np.array([1, 1, 2, 2, 3, 3])
    folds = module.leave_one_group_out_folds(pair_ids)

    assert len(folds) == 3
    tested = []
    for train, test in folds:
        assert set(pair_ids[train]).isdisjoint(set(pair_ids[test]))
        assert len(np.unique(pair_ids[test])) == 1
        tested.extend(test.tolist())
    assert sorted(tested) == list(range(len(pair_ids)))


def test_outer_fold_selects_units_without_seeing_test_features() -> None:
    """Catch unit selection performed on the combined train and test data."""
    module = analysis_module()
    x_train = np.array(
        [
            [-2.0, 0.0],
            [-1.8, 0.2],
            [-1.6, -0.2],
            [1.6, 0.0],
            [1.8, 0.2],
            [2.0, -0.2],
        ]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[0.0, -1000.0], [0.0, 1000.0]])

    probability, chosen_c, selected = module.fit_outer_fold(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        k=1,
        regularization="fixed",
        fixed_c=0.1,
        candidate_cs=(0.01, 0.1, 1.0),
        random_state=7,
    )

    np.testing.assert_array_equal(selected, np.array([0]))
    assert chosen_c == 0.1
    assert probability.shape == (2,)


def test_repeated_cv_scores_each_trial_after_probability_aggregation() -> None:
    """Catch an analysis that scores repeated outer-fold rows as independent trials."""
    module = analysis_module()
    y = np.tile(np.array([0, 1]), 10)
    x = np.column_stack(
        [
            np.where(y == 1, 2.0, -2.0) + np.linspace(-0.1, 0.1, len(y)),
            np.linspace(-1.0, 1.0, len(y)),
        ]
    )
    result = module.cross_validated_run(
        x=x,
        y=y,
        pair_ids=np.repeat(np.arange(10), 2),
        k=1,
        validation="all_sentences",
        regularization="fixed",
        fixed_c=0.1,
        candidate_cs=(0.01, 0.1, 1.0),
        n_repeats=3,
        random_state=19,
    )

    np.testing.assert_array_equal(result["prediction_count"], np.full(len(y), 3))
    assert result["predictive_information_bits"] > 0
    assert result["auc"] > 0.95


def test_sentence_pair_cv_predicts_each_trial_once() -> None:
    """Catch accidental repeated or incomplete scoring in leave-pair-out CV."""
    module = analysis_module()
    y = np.tile(np.array([0, 1]), 5)
    x = np.column_stack([np.where(y == 1, 1.0, -1.0), np.arange(len(y))])
    result = module.cross_validated_run(
        x=x,
        y=y,
        pair_ids=np.repeat(np.arange(5), 2),
        k=1,
        validation="unseen_sentence_pair",
        regularization="fixed",
        fixed_c=0.1,
        candidate_cs=(0.01, 0.1, 1.0),
        n_repeats=99,
        random_state=23,
    )

    np.testing.assert_array_equal(result["prediction_count"], np.ones(len(y), dtype=int))
    assert result["predictive_information_bits"] > 0


def test_nested_regularization_is_selected_from_training_candidates() -> None:
    """Catch bypassing the declared inner-CV regularization grid."""
    module = analysis_module()
    y_train = np.tile(np.array([0, 1]), 8)
    x_train = np.column_stack(
        [
            np.where(y_train == 1, 1.5, -1.5) + np.linspace(-0.2, 0.2, 16),
            np.sin(np.arange(16)),
        ]
    )
    x_test = np.array([[-1.0, 0.0], [1.0, 0.0]])
    candidates = (0.01, 0.1, 1.0)

    first = module.fit_outer_fold(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        k=1,
        regularization="nested",
        fixed_c=99.0,
        candidate_cs=candidates,
        random_state=31,
    )
    second = module.fit_outer_fold(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        k=1,
        regularization="nested",
        fixed_c=99.0,
        candidate_cs=candidates,
        random_state=31,
    )

    assert first[1] in candidates
    assert first[1] != 99.0
    assert first[1] == second[1]
    np.testing.assert_allclose(first[0], second[0])


def test_exact_patient_sign_test_uses_patients_and_excludes_ties() -> None:
    """Catch run-level pseudoreplication or counting exact-zero ties as successes."""
    module = analysis_module()
    result = module.exact_sign_test(np.array([0.2, 0.1, 0.3, 0.4, 0.5, 0.0]))

    assert result == {
        "n_nonzero": 5,
        "n_positive": 5,
        "p_greater": 0.03125,
    }
