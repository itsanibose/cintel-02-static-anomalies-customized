"""utils_clinic.py.

Shared helper functions for the clinic anomaly-detection domain.

Used by:
- src/cintel/anomaly_detector.py (cintel-02)
- app.py
"""

from pathlib import Path
from typing import Final

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import polars as pl

# === DECLARE GLOBAL CONSTANTS FOR FOLDER PATHS (directories) ===
ROOT_DIR: Final[Path] = Path.cwd()
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"


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
    """Return rows meeting age/height outlier threshold conditions."""
    df = (
        df.with_columns(
            (pl.col("age_years") / pl.col("height_inches")).alias("age_height_ratio")
        )
        .with_columns(
            pl.when(
                (pl.col("age_height_ratio") < 0.03)
                | (pl.col("age_height_ratio") > 0.30)
            )
            .then(pl.lit("outlier"))
            .otherwise(pl.lit("normal"))
            .alias("outlier_flag")
        )
        .with_columns(
            (
                (pl.col("age_height_ratio") - pl.col("age_height_ratio").mean())
                / pl.col("age_height_ratio").std()
            ).alias("age_height_zscore")
        )
    )

    if include == "yes":
        return df

    return df.filter(pl.col("outlier_flag") == "outlier")


# Adding function for a scatter plot of age vs height
def build_age_height_figure(df: pl.DataFrame) -> Figure:
    """Build and return a scatter plot figure of age vs height.

    Used directly by app.py (marimo renders the returned figure).
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pass Polars columns directly into the scatter function
    ax.scatter(
        x=df["age_years"],
        y=df["height_inches"],
        color="crimson",
        alpha=0.6,
        edgecolors="black",
    )

    # Customizing the chart elements
    ax.set_title(
        "Age vs. Height Scatter Plot (Using Polars)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Age (Years)", fontsize=11)
    ax.set_ylabel("Height (Inches)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def plot_age_height(df: pl.DataFrame) -> None:
    """Show the age vs height scatter plot and save it to artifacts/."""
    fig = build_age_height_figure(df)

    plt.show()
    fig.savefig(ARTIFACTS_DIR / "age_vs_height_plot.png", dpi=300, bbox_inches="tight")
