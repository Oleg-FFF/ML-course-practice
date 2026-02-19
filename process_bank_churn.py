# process_bank_churn.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "Exited"
DEFAULT_DROP_COLS = ["id", "CustomerId", "Surname", TARGET_COL]


@dataclass(frozen=True)
class PreprocessArtifacts:
    """
    Container for fitted preprocessing objects and metadata required to process new data.
    """
    input_cols: List[str]
    numeric_cols: List[str]
    geo_col: str
    gender_col: str
    scaler_numeric: bool
    scaler: Optional[StandardScaler]
    encoder: OneHotEncoder
    feature_names: List[str]


def _get_input_cols(raw_df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
    """
    Select input feature columns by excluding drop_cols.

    Args:
        raw_df: Raw input dataframe.
        drop_cols: Columns to exclude from features.

    Returns:
        List of feature column names.
    """
    return [c for c in raw_df.columns if c not in drop_cols]


def _split_train_val(
    raw_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split raw dataframe into train/validation parts with stratification by target.

    Args:
        raw_df: Raw dataframe containing target.
        target_col: Target column name.
        test_size: Validation share.
        random_state: Random seed.

    Returns:
        (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col],
    )
    return train_df, val_df


def _gender_to_binary(series: pd.Series) -> np.ndarray:
    """
    Convert Gender to binary feature: Female -> 1, else -> 0.

    Args:
        series: Pandas Series with gender strings.

    Returns:
        2D numpy array shape (n_samples, 1) with 0/1 values.
    """
    return (series.astype(str) == "Female").astype(int).to_numpy().reshape(-1, 1)


def _fit_encoder(train_df: pd.DataFrame, geo_col: str) -> OneHotEncoder:
    """
    Fit OneHotEncoder on Geography column.

    Args:
        train_df: Training dataframe.
        geo_col: Geography column name.

    Returns:
        Fitted OneHotEncoder.
    """
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(train_df[[geo_col]])
    return enc


def _fit_scaler(train_df: pd.DataFrame, numeric_cols: List[str]) -> StandardScaler:
    """
    Fit StandardScaler on numeric columns.

    Args:
        train_df: Training dataframe.
        numeric_cols: List of numeric feature columns.

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    return scaler


def _build_feature_names(
    encoder: OneHotEncoder,
    geo_col: str,
    numeric_cols: List[str],
) -> List[str]:
    """
    Build final feature names in the exact order used in X matrix.

    Order:
      1) gender_binary
      2) one-hot geography
      3) numeric columns (scaled or raw)

    Args:
        encoder: Fitted OneHotEncoder for geo.
        geo_col: Geography column name.
        numeric_cols: Numeric column names.

    Returns:
        List of feature names in correct order.
    """
    geo_names = encoder.get_feature_names_out([geo_col]).tolist()
    return ["Gender_is_Female"] + geo_names + numeric_cols


def _transform_df_to_matrix(
    df: pd.DataFrame,
    *,
    artifacts: PreprocessArtifacts,
) -> np.ndarray:
    """
    Transform dataframe into model-ready numeric matrix X using fitted artifacts.

    Args:
        df: Input dataframe containing at least artifacts.input_cols.
        artifacts: PreprocessArtifacts with fitted scaler/encoder and column info.

    Returns:
        X as numpy array of shape (n_samples, n_features).
    """
    # 1) Gender -> binary
    gender_bin = _gender_to_binary(df[artifacts.gender_col])

    # 2) Geography -> OHE
    geo_ohe = artifacts.encoder.transform(df[[artifacts.geo_col]])

    # 3) Numeric -> scaled or raw
    numeric = df[artifacts.numeric_cols]
    if artifacts.scaler_numeric:
        numeric = artifacts.scaler.transform(numeric)
    else:
        numeric = numeric.to_numpy()

    # concat
    X = np.hstack([gender_bin, geo_ohe, numeric])
    return X


def preprocess_data(
    raw_df: pd.DataFrame,
    *,
    scaler_numeric: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: Optional[List[str]] = None,
    geo_col: str = "Geography",
    gender_col: str = "Gender",
    target_col: str = TARGET_COL,
) -> tuple[ndarray, Any, ndarray, Any, list[str], StandardScaler | None, OneHotEncoder, list[str]]:
    """
    Preprocess raw churn dataframe into train/val matrices and fitted preprocessors.

    Steps:
      1) Choose columns (drop Surname by default).
      2) Split into train/validation (stratified).
      3) One-hot encode Geography (handle_unknown=ignore).
      4) Convert Gender to binary (Female=1).
      5) Optionally scale numeric features.

    Args:
        raw_df: Raw dataset dataframe containing features and target.
        scaler_numeric: If True, apply StandardScaler to numeric features.
        test_size: Validation fraction.
        random_state: Random seed.
        drop_cols: Columns to exclude from input features. If None, uses DEFAULT_DROP_COLS.
        geo_col: Name of geography categorical column.
        gender_col: Name of gender column.
        target_col: Name of target column.

    Returns:
        X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder
    """
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    # Split
    train_df, val_df = _split_train_val(
        raw_df=raw_df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    # Define inputs/targets
    input_cols = _get_input_cols(raw_df, drop_cols)
    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()

    # Numeric cols = all numeric among inputs EXCEPT we will treat Gender/Geography separately
    numeric_cols = train_inputs.select_dtypes(include="number").columns.tolist()
    # Guard in case Gender/Geography got inferred as numeric (rare) — ensure they are not in numeric list
    numeric_cols = [c for c in numeric_cols if c not in {geo_col, gender_col}]

    # Fit encoder/scaler on TRAIN only
    encoder = _fit_encoder(train_inputs, geo_col=geo_col)

    scaler: Optional[StandardScaler]
    if scaler_numeric:
        scaler = _fit_scaler(train_inputs, numeric_cols=numeric_cols)
    else:
        scaler = None

    # Build artifacts for consistent transform
    feature_names = _build_feature_names(encoder, geo_col=geo_col, numeric_cols=numeric_cols)
    artifacts = PreprocessArtifacts(
        input_cols=input_cols,
        numeric_cols=numeric_cols,
        geo_col=geo_col,
        gender_col=gender_col,
        scaler_numeric=scaler_numeric,
        scaler=scaler,
        encoder=encoder,
        feature_names=feature_names,
    )

    # Transform train/val
    X_train = _transform_df_to_matrix(train_inputs, artifacts=artifacts)
    X_val = _transform_df_to_matrix(val_inputs, artifacts=artifacts)

    return (
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder,
        feature_names
    )


def preprocess_new_data(
    new_df: pd.DataFrame,
    *,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder,
    scaler_numeric: bool = True,
    geo_col: str = "Geography",
    gender_col: str = "Gender",
) -> np.ndarray:
    """
    Preprocess new/unseen data (e.g., test.csv) using already-fitted scaler and encoder.

    Important: new_df must contain the same input_cols as training (except target).

    Args:
        new_df: New data dataframe.
        input_cols: Feature columns used in training.
        scaler: Fitted StandardScaler (or None if scaler_numeric=False).
        encoder: Fitted OneHotEncoder for Geography.
        scaler_numeric: Whether to scale numeric features.
        geo_col: Geography column name.
        gender_col: Gender column name.

    Returns:
        X matrix ready for model inference.
    """
    inputs = new_df[input_cols].copy()
    numeric_cols = inputs.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {geo_col, gender_col}]

    artifacts = PreprocessArtifacts(
        input_cols=input_cols,
        numeric_cols=numeric_cols,
        geo_col=geo_col,
        gender_col=gender_col,
        scaler_numeric=scaler_numeric,
        scaler=scaler,
        encoder=encoder,
        feature_names=_build_feature_names(encoder, geo_col=geo_col, numeric_cols=numeric_cols),
    )
    return _transform_df_to_matrix(inputs, artifacts=artifacts)