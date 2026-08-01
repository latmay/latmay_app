from __future__ import annotations

import pandas as pd


CLEARANCE_COLUMNS = [
    "requires_clearance",
    "clearance_type",
    "clearance_requirement_type",
    "clearance_evidence_text",
]


def _truthy_clearance_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if not text:
        return False
    if text in {"false", "0", "no", "none", "null", "nan"}:
        return False
    return True


def filter_jobs_df_excluding_security_clearance(
    jobs_df: pd.DataFrame,
    *,
    exclude_security_clearance: bool,
) -> pd.DataFrame:
    if not exclude_security_clearance:
        return jobs_df.copy()

    available_columns = [column for column in CLEARANCE_COLUMNS if column in jobs_df.columns]
    if not available_columns:
        return jobs_df.copy()

    clearance_mask = pd.Series(False, index=jobs_df.index)
    for column in available_columns:
        clearance_mask = clearance_mask | jobs_df[column].map(_truthy_clearance_value)

    return jobs_df.loc[~clearance_mask].copy()
