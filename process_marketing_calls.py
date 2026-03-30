# process_marketing_calls.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "y"
DEFAULT_DROP_COLS = ["y", "deposit_subscribed", "duration"]


@dataclass(frozen=True)
class PreprocessArtifacts:
    input_cols: List[str]
    numeric_cols: List[str]
    categorical_ohe_cols: List[str]
    binary_cols: List[str]

    education_col: str
    default_col: str

    scaler_numeric: bool
    scaler: Optional[StandardScaler]
    encoder: OneHotEncoder
    feature_names: List[str]


def _get_input_cols(raw_df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
    return [c for c in raw_df.columns if c not in drop_cols]


def _split_train_val(
    raw_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col],
    )
    return train_df, val_df


def _prepare_target(series: pd.Series) -> pd.Series:
    """
    Convert target y to binary:
    yes -> 1, no -> 0
    """
    if series.dtype == object:
        return series.map({"yes": 1, "no": 0}).astype(int)
    return series.astype(int)


def _process_education(series: pd.Series) -> pd.Series:
    """
    Group education into more stable buckets:
    low / medium / high / unknown
    """
    def map_education(x: Any) -> str:
        x = str(x)
        if x in {"illiterate", "basic.4y", "basic.6y", "basic.9y"}:
            return "low"
        if x == "high.school":
            return "medium"
        if x in {"professional.course", "university.degree"}:
            return "high"
        return "unknown"

    return series.apply(map_education)


def _process_default(series: pd.Series) -> pd.Series:
    """
    default:
    no -> 0
    unknown/yes -> 1
    """
    return series.map({"no": 0, "unknown": 1, "yes": 1}).fillna(1).astype(int)


def _yes_or_no_to_binary(series: pd.Series) -> pd.Series:
    """
    yes -> 1, anything else -> 0
    Useful for housing / loan
    """
    return (series.astype(str) == "yes").astype(int)


def _fit_encoder(train_df: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(train_df[categorical_cols])
    return enc


def _fit_scaler(train_df: pd.DataFrame, numeric_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    return scaler


def _build_feature_names(
    encoder: OneHotEncoder,
    categorical_cols: List[str],
    binary_cols: List[str],
    numeric_cols: List[str],
) -> List[str]:
    ohe_names = encoder.get_feature_names_out(categorical_cols).tolist()
    return ohe_names + binary_cols + numeric_cols


def _prepare_inputs(
    df: pd.DataFrame,
    *,
    education_col: str,
    default_col: str,
    binary_cols: List[str],
) -> pd.DataFrame:
    """
    Create transformed intermediate dataframe before OHE/scaling.
    """
    prepared = df.copy()

    prepared[education_col] = _process_education(prepared[education_col])
    prepared[default_col] = _process_default(prepared[default_col])

    for col in binary_cols:
        prepared[col] = _yes_or_no_to_binary(prepared[col])

    return prepared


def _transform_df_to_matrix(
    df: pd.DataFrame,
    *,
    artifacts: PreprocessArtifacts,
) -> np.ndarray:
    prepared = _prepare_inputs(
        df,
        education_col=artifacts.education_col,
        default_col=artifacts.default_col,
        binary_cols=artifacts.binary_cols,
    )

    ohe_part = artifacts.encoder.transform(prepared[artifacts.categorical_ohe_cols])

    binary_part = prepared[artifacts.binary_cols].to_numpy(dtype=float)

    numeric_part = prepared[artifacts.numeric_cols]
    if artifacts.scaler_numeric and artifacts.scaler is not None:
        numeric_part = artifacts.scaler.transform(numeric_part)
    else:
        numeric_part = numeric_part.to_numpy(dtype=float)

    X = np.hstack([ohe_part, binary_part, numeric_part])
    return X


def preprocess_data(
    raw_df: pd.DataFrame,
    *,
    scaler_numeric: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: Optional[List[str]] = None,
    job_col: str = "job",
    marital_col: str = "marital",
    education_col: str = "education",
    default_col: str = "default",
    housing_col: str = "housing",
    loan_col: str = "loan",
    contact_col: str = "contact",
    month_col: str = "month",
    day_of_week_col: str = "day_of_week",
    poutcome_col: str = "poutcome",
    target_col: str = TARGET_COL,
    use_month: bool = True,
    use_day_of_week: bool = True,
) -> tuple[
    ndarray,
    pd.Series,
    ndarray,
    pd.Series,
    List[str],
    Optional[StandardScaler],
    OneHotEncoder,
    List[str],
    PreprocessArtifacts,
]:
    """
    Preprocess bank marketing dataset into train/validation matrices.

    Notes:
    - duration is dropped by default
    - education is grouped into stable buckets
    - default is transformed into binary
    - housing and loan are binary
    - categorical features are OHE
    """
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS.copy()

    train_df, val_df = _split_train_val(
        raw_df=raw_df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    input_cols = _get_input_cols(raw_df, drop_cols)

    train_targets = _prepare_target(train_df[target_col].copy())
    val_targets = _prepare_target(val_df[target_col].copy())

    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()

    categorical_ohe_cols = [
        job_col,
        marital_col,
        education_col,
        contact_col,
        poutcome_col,
    ]

    if use_month:
        categorical_ohe_cols.append(month_col)

    if use_day_of_week:
        categorical_ohe_cols.append(day_of_week_col)

    binary_cols = [housing_col, loan_col]
    transformed_binary_cols = binary_cols + [default_col]

    train_prepared = _prepare_inputs(
        train_inputs,
        education_col=education_col,
        default_col=default_col,
        binary_cols=binary_cols,
    )

    val_prepared = _prepare_inputs(
        val_inputs,
        education_col=education_col,
        default_col=default_col,
        binary_cols=binary_cols,
    )

    numeric_cols = train_prepared.select_dtypes(include="number").columns.tolist()
    numeric_cols = [
        c for c in numeric_cols
        if c not in transformed_binary_cols
    ]

    encoder = _fit_encoder(train_prepared, categorical_cols=categorical_ohe_cols)

    scaler: Optional[StandardScaler]
    if scaler_numeric and numeric_cols:
        scaler = _fit_scaler(train_prepared, numeric_cols=numeric_cols)
    else:
        scaler = None

    feature_names = _build_feature_names(
        encoder=encoder,
        categorical_cols=categorical_ohe_cols,
        binary_cols=transformed_binary_cols,
        numeric_cols=numeric_cols,
    )

    artifacts = PreprocessArtifacts(
        input_cols=input_cols,
        numeric_cols=numeric_cols,
        categorical_ohe_cols=categorical_ohe_cols,
        binary_cols=binary_cols,
        education_col=education_col,
        default_col=default_col,
        scaler_numeric=scaler_numeric,
        scaler=scaler,
        encoder=encoder,
        feature_names=feature_names,
    )

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
        feature_names,
        artifacts,
    )


def preprocess_new_data(
    new_df: pd.DataFrame,
    *,
    artifacts: PreprocessArtifacts,
) -> np.ndarray:
    """
    Preprocess new data using already fitted artifacts.
    """
    inputs = new_df[artifacts.input_cols].copy()
    return _transform_df_to_matrix(inputs, artifacts=artifacts)