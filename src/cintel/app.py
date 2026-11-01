"""app.py - Reactive marimo app for this project.

Marimo adds reactive controls.
Every function below is a marimo cell.

RUN LOCALLY:

  uv run marimo edit src/cintel/app.py     # edit
  uv run marimo run  src/cintel/app.py     # run as an app (CTRL+c quits)

WASM DEPENDENCIES:

Under WebAssembly (emscripten) the browser has no project environment.
This app uses polars and matplotlib in addition to marimo, so the first
cell installs them with micropip when running under WASM.

NO LOGGING:

A browser-based WASM app has no persistent Python server to store log files,
so this notebook configures no logging.

PLAN CELLS FIRST

1. Imports (installs WASM deps, returns shared imports and constants)
2. Opening title and introduction (Markdown)
3. Load the clinic data (same cell handles local and WASM environments)
4. Controls (anomaly threshold sliders + outlier radio)
5. Detect anomalies (same logic as the CLI pipeline, via cintel.utils_clinic)
6. Detect outliers (age/height ratio, via cintel.utils_clinic)
7. Show the results side by side (anomalies left, outliers right)
8. Age vs height scatter plot (below the two tables)
9. Closing (Markdown)

Note: marimo triggers @app.cell functions automatically; we never call them.
The names starting with "_" are for our own organization; the engine ignores
them.
"""

# === ONLY THIS AT THE TOP OF THE FILE ===

import marimo

__generated_with = "0.25.0"

app = marimo.App(width="medium")

# === FIRST CELL IMPORTS AND RETURNS DEPENDENCIES TO MAKE IT SELF-CONTAINED ===


@app.cell
async def _():
    import sys

    if sys.platform == "emscripten":
        import micropip

        await micropip.install(["polars", "matplotlib"])

    import marimo as mo
    import polars as pl

    from cintel.utils_clinic import (
        build_age_height_figure,
        detect_clinic_anomalies,
        detect_outliers,
        load_clinic_data,
    )

    return (
        build_age_height_figure,
        detect_clinic_anomalies,
        detect_outliers,
        load_clinic_data,
        mo,
        pl,
        sys,
    )


# === TITLE AND INTRODUCTION ===


@app.cell
def _(mo):
    mo.md(r"""
    # Static Anomaly and Outlier Detection

    This app flags pediatric clinic records where **age** or **height** or ratio of age to height
    is higher than expected.

    Move the sliders below to change what counts as an anomaly.
    """)


# === LOAD THE CLINIC DATA ===


@app.cell
def _(load_clinic_data, mo, sys):
    """Load the clinic data. Two environments, one cell.

    Locally, polars reads the CSV from data/.
    Under WASM (emscripten), Pyodide's open_url() fetches the CSV from
    public/ and load_clinic_data() reads it from there instead.
    """
    notebook_location = mo.notebook_location()

    if notebook_location is None:
        raise RuntimeError("Unable to determine notebook location.")

    filename = "clinic_data.csv"

    if sys.platform == "emscripten":
        from pyodide.http import open_url  # ty: ignore[unresolved-import]

        data_url = notebook_location / "public" / filename
        df = load_clinic_data(open_url(str(data_url)))
    else:
        data_path = notebook_location.parents[1] / "data" / filename
        df = load_clinic_data(data_path)

    return (df,)


# === CONTROLS (DISPLAYED IN THE LAYOUT CELL BELOW) ===


@app.cell
def _(mo):
    max_age = mo.ui.slider(0, 20, value=15, label="Max reasonable age (years)")
    max_height = mo.ui.slider(0, 90, value=70, label="Max reasonable height (inches)")
    include_all = mo.ui.radio(
        options={"All records (with outlier flag)": "yes", "Outliers only": "no"},
        value="All records (with outlier flag)",
        label="Records to show in the outlier table",
    )
    return include_all, max_age, max_height


# === DETECT ANOMALIES (SAME LOGIC AS THE CLI PIPELINE) ===


@app.cell
def _(detect_clinic_anomalies, df, max_age, max_height):
    anomalies_df = detect_clinic_anomalies(
        df,
        max_age_years=max_age.value,
        max_height_inches=max_height.value,
    )
    return (anomalies_df,)


# === DETECT OUTLIERS ===


@app.cell
def _(detect_outliers, df, include_all):
    outliers_df = detect_outliers(df, include=include_all.value)
    return (outliers_df,)


# === SHOW THE RESULTS SIDE BY SIDE ===


@app.cell
def _(anomalies_df, df, include_all, max_age, max_height, mo, outliers_df, pl):
    _n_outliers = outliers_df.filter(pl.col("outlier_flag") == "outlier").height

    _anomaly_panel = mo.vstack(
        [
            mo.md("### Anomalies (age / height thresholds)"),
            mo.hstack([max_age, max_height], justify="start", gap=2),
            mo.md(
                f"**{anomalies_df.height} of {df.height} records** flagged as anomalies."
            ),
            mo.ui.table(anomalies_df),
        ]
    )

    _outlier_panel = mo.vstack(
        [
            mo.md("### Outliers (age / height ratio)"),
            include_all,
            mo.md(
                f"**{_n_outliers} of {df.height} records** flagged as outliers "
                "(age/height ratio below 0.03 or above 0.30)."
            ),
            mo.ui.table(outliers_df),
        ]
    )

    mo.hstack([_anomaly_panel, _outlier_panel], widths="equal", gap=2, align="start")


# === AGE VS HEIGHT SCATTER PLOT ===


@app.cell
def _(build_age_height_figure, df, mo):
    mo.vstack(
        [
            mo.md("### Age vs. Height"),
            build_age_height_figure(df),
        ]
    )


# === TYPICALLY END WITH A MARKDOWN SOURCE LINK AND CLOSING ===


@app.cell
def _(mo):
    mo.md(r"""
    ---
    [😊 Learn more at marimo.io](https://marimo.io/)
    ---

    **Running locally?**
    To stop the app, click in the VS Code terminal,
    then press **CTRL** and **c** together (CTRL+c).
    When it asks "Are you sure you want to quit? (y/N)",
    type **y** and press **Enter**.
    ---
    """)


if __name__ == "__main__":
    app.run()
