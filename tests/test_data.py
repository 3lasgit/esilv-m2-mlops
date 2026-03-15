"""Unit tests for src/data.py — preprocessing helpers."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from data import (
    CAT_TO_ENCODE,
    COLUMNS,
    OUTLIER_COLS,
    _add_features,
    _cap_outliers_iqr,
    _encode,
    _load_raw,
    load_and_preprocess,
)

# ── _cap_outliers_iqr ────────────────────────────────────────


class TestCapOutliersIqr:
    def test_clips_extreme_values(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        result = _cap_outliers_iqr(df, ["x"], factor=1.5)
        assert result["x"].max() < 100

    def test_does_not_modify_original(self, sample_df):
        original = sample_df.copy()
        _cap_outliers_iqr(sample_df, OUTLIER_COLS)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_preserves_shape(self, sample_df):
        result = _cap_outliers_iqr(sample_df, OUTLIER_COLS)
        assert result.shape == sample_df.shape

    def test_values_within_iqr_bounds(self):
        df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
        result = _cap_outliers_iqr(df, ["v"], factor=1.5)
        q1 = df["v"].quantile(0.25)
        q3 = df["v"].quantile(0.75)
        iqr = q3 - q1
        assert result["v"].max() <= q3 + 1.5 * iqr
        assert result["v"].min() >= q1 - 1.5 * iqr


# ── _add_features ────────────────────────────────────────────


class TestAddFeatures:
    def test_adds_three_columns(self, sample_df):
        result = _add_features(sample_df)
        new_cols = {"hr_age_ratio", "cardio_risk_score", "exang_oldpeak"}
        assert new_cols.issubset(set(result.columns))

    def test_hr_age_ratio_computation(self, sample_df):
        result = _add_features(sample_df)
        expected = sample_df["thalach"] / sample_df["age"]
        np.testing.assert_array_almost_equal(result["hr_age_ratio"], expected)

    def test_cardio_risk_score_computation(self, sample_df):
        result = _add_features(sample_df)
        expected = (
            (sample_df["trestbps"] / 120) * (sample_df["chol"] / 200) * (sample_df["age"] / 60)
        )
        np.testing.assert_array_almost_equal(result["cardio_risk_score"], expected)

    def test_exang_oldpeak_computation(self, sample_df):
        result = _add_features(sample_df)
        expected = sample_df["exang"] * sample_df["oldpeak"]
        np.testing.assert_array_almost_equal(result["exang_oldpeak"], expected)

    def test_does_not_modify_original(self, sample_df):
        original = sample_df.copy()
        _add_features(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)


# ── _encode ──────────────────────────────────────────────────


class TestEncode:
    def test_removes_original_cat_columns(self, sample_df):
        result = _encode(sample_df)
        for col in CAT_TO_ENCODE:
            assert col not in result.columns

    def test_creates_dummy_columns(self, sample_df):
        result = _encode(sample_df)
        dummy_cols = [
            c for c in result.columns if any(c.startswith(f"{cat}_") for cat in CAT_TO_ENCODE)
        ]
        assert len(dummy_cols) > 0

    def test_preserves_non_cat_columns(self, sample_df):
        non_cat = [c for c in sample_df.columns if c not in CAT_TO_ENCODE]
        result = _encode(sample_df)
        for col in non_cat:
            assert col in result.columns

    def test_does_not_modify_original(self, sample_df):
        original = sample_df.copy()
        _encode(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_row_count_preserved(self, sample_df):
        result = _encode(sample_df)
        assert len(result) == len(sample_df)


# ── Constants ────────────────────────────────────────────────


class TestConstants:
    def test_columns_has_target(self):
        assert "target" in COLUMNS

    def test_columns_count(self):
        assert len(COLUMNS) == 14

    def test_outlier_cols_subset_of_columns(self):
        assert all(c in COLUMNS for c in OUTLIER_COLS)

    def test_cat_to_encode_subset_of_columns(self):
        assert all(c in COLUMNS for c in CAT_TO_ENCODE)


# ── _load_raw ────────────────────────────────────────────────


class TestLoadRaw:
    def test_loads_dataframe_with_target(self, sample_df):
        with patch("data.pd.read_csv", return_value=sample_df):
            df = _load_raw()
            assert "target" in df.columns
            assert isinstance(df, pd.DataFrame)

    def test_fallback_adds_columns_if_no_target(self, sample_df):
        df_no_target = sample_df.rename(columns={"target": "other"})
        with patch("data.pd.read_csv", return_value=df_no_target):
            df = _load_raw()
            assert "target" in df.columns

    def test_fallback_on_csv_failure(self, sample_df):
        with (
            patch("data.pd.read_csv", side_effect=Exception("network error")),
            patch("sklearn.datasets.fetch_openml") as mock_openml,
        ):
            mock_openml.return_value.frame = sample_df.copy()
            mock_openml.return_value.frame.rename(columns={"target": "class"}, inplace=True)
            mock_openml.return_value.frame["class"] = "present"
            df = _load_raw()
            assert "target" in df.columns


# ── load_and_preprocess ──────────────────────────────────────


class TestLoadAndPreprocess:
    def test_returns_correct_tuple_length(self, sample_df):
        with patch("data._load_raw", return_value=sample_df):
            result = load_and_preprocess()
            assert len(result) == 8

    def test_shapes_consistent(self, sample_df):
        with patch("data._load_raw", return_value=sample_df):
            X_train, X_test, X_train_s, X_test_s, y_train, y_test, feat, scaler = (
                load_and_preprocess()
            )
            assert X_train.shape[0] == len(y_train)
            assert X_test.shape[0] == len(y_test)
            assert X_train_s.shape == X_train.shape
            assert X_test_s.shape == X_test.shape

    def test_feature_names_match_columns(self, sample_df):
        with patch("data._load_raw", return_value=sample_df):
            X_train, _, _, _, _, _, feat, _ = load_and_preprocess()
            assert feat == list(X_train.columns)

    def test_scaler_is_fitted(self, sample_df):
        with patch("data._load_raw", return_value=sample_df):
            _, _, _, _, _, _, _, scaler = load_and_preprocess()
            assert hasattr(scaler, "mean_")
