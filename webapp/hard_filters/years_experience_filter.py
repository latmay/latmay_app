from __future__ import annotations

"""
Filter jobs by an already-computed minimum years-of-experience column.

This module does not run regex extraction. The data pipeline is responsible for
populating min_years_experience before ranking.
"""

import pandas as pd


def filter_jobs_df_by_max_required_yoe(
    jobs_df: pd.DataFrame,
    max_required_yoe: float | None,
    yoe_column: str = "min_years_experience",
) -> pd.DataFrame:
    if max_required_yoe is None:
        return jobs_df.copy()

    if yoe_column not in jobs_df.columns:
        raise ValueError(
            f"Column '{yoe_column}' not found in dataframe. "
            f"Available columns: {list(jobs_df.columns)}"
        )

    yoe = pd.to_numeric(jobs_df[yoe_column], errors="coerce")
    mask = yoe.isna() | (yoe <= max_required_yoe)

    return jobs_df.loc[mask].copy()
