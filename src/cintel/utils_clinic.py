"""utils_clinic.py.

Shared helper functions for the clinic anomaly-detection domain.

Used by:
- src/cintel/anomaly_detector.py (cintel-02)
- app.py
"""

from pathlib import Path

import polars as pl


def load_clinic_data(data_file: Path) -> pl.DataFrame:
    """Read pediatric clinic age/height records from a CSV file."""
    return pl.read_csv(data_file)


def detect_clinic_anomalies(
    df: pl.DataFrame,
    max_age_years: float,
    max_height_inches: float,
) -> pl.DataFrame:
    """Return rows meeting one or more defined anomaly threshold conditions."""
    return df.filter(
        (pl.col("age_years") >= max_age_years)
        | (pl.col("height_inches") >= max_height_inches)
    )


def detect_outliers(df: pl.DataFrame, include: str = "yes") -> pl.DataFrame:
    """Return rows meeting one or more defined outlier threshold conditions."""
    df = df.with_columns(
        (pl.col("age_years") / pl.col("height_inches")).alias("age_height_ratio")
    ).with_columns(
        pl.when(
            (pl.col("age_height_ratio") < 0.03) | (pl.col("age_height_ratio") > 0.30)
        )
        .then(pl.lit("outlier"))
        .otherwise(pl.lit("normal"))
        .alias("outlier_flag")
    )
    if include == "yes":
        return df

    return df.filter(pl.col("outlier_flag") == "outlier")
