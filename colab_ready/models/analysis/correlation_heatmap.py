import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _find_column(df: pd.DataFrame, aliases: List[str]) -> str:
    normalized = {_normalize(c): c for c in df.columns}
    for alias in aliases:
        key = _normalize(alias)
        if key in normalized:
            return normalized[key]
    return ""


def _load_single_nsrdb_csv(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    metadata_df = pd.read_csv(path, nrows=1)
    metadata = {str(k): str(v) for k, v in metadata_df.iloc[0].to_dict().items()}
    data_df = pd.read_csv(path, skiprows=2)
    return data_df, metadata


def load_nsrdb(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
    else:
        csv_files = [path]

    frames = []
    for csv_file in csv_files:
        df, _ = _load_single_nsrdb_csv(csv_file)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    aliases = {
        "ghi": ["GHI", "Global Horizontal Irradiance"],
        "temperature": ["Temperature", "Air Temperature", "Temp"],
        "relative_humidity": ["Relative Humidity", "RH", "Humidity"],
        "wind_speed": ["Wind Speed", "WindSpeed"],
        "pressure": ["Pressure", "Surface Pressure", "Atmospheric Pressure"],
        "solar_zenith_angle": ["Solar Zenith Angle", "Zenith Angle", "Solar Zenith"],
    }

    selected = {}
    for out_name, options in aliases.items():
        selected[out_name] = _find_column(raw, options)

    required = ["ghi", "temperature", "relative_humidity", "wind_speed", "pressure"]
    missing = [k for k in required if not selected[k]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(raw.columns)}")

    if not all(col in raw.columns for col in ["Year", "Month", "Day", "Hour", "Minute"]):
        raise ValueError("NSRDB data must include Year, Month, Day, Hour, Minute.")

    timestamp = pd.to_datetime(
        {
            "year": raw["Year"],
            "month": raw["Month"],
            "day": raw["Day"],
            "hour": raw["Hour"],
            "minute": raw["Minute"],
        },
        errors="coerce",
    )

    df = pd.DataFrame({"timestamp": timestamp})
    for out_name, col_name in selected.items():
        if col_name:
            df[out_name] = pd.to_numeric(raw[col_name], errors="coerce")

    hour = df["timestamp"].dt.hour.astype(float)
    doy = df["timestamp"].dt.dayofyear.astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    df["ghi_lag_24"] = df["ghi"].shift(24)
    df["ghi_roll_mean_3"] = df["ghi"].rolling(window=3).mean()
    df["ghi_roll_mean_6"] = df["ghi"].rolling(window=6).mean()
    df["ghi_roll_mean_12"] = df["ghi"].rolling(window=12).mean()

    return df.dropna().reset_index(drop=True)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    corr_df = df.select_dtypes(include=[np.number]).copy()
    corr_matrix = corr_df.corr(method="pearson")

    labels = list(corr_matrix.columns)
    matrix = corr_matrix.to_numpy()

    _ensure_parent_dir(output_path)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson r")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Pearson Correlation Heatmap: Meteorological Features vs GHI")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return corr_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Pearson correlation heatmap for NSRDB meteorological data and GHI")
    parser.add_argument("--data-path", type=str, default="dataset", help="Path to NSRDB CSV file or folder")
    parser.add_argument("--output", type=str, default="outputs/plots/analysis/pearson_correlation_heatmap.png", help="Output image path")
    args = parser.parse_args()

    df = load_nsrdb(args.data_path)
    corr_matrix = plot_correlation_heatmap(df, args.output)

    print(f"Saved heatmap to: {args.output}")

    if "ghi" in corr_matrix.columns:
        ghi_corr = corr_matrix["ghi"].drop(labels=["ghi"]).sort_values(key=np.abs, ascending=False)
        print("\nTop correlations with GHI (absolute Pearson r):")
        for feature, value in ghi_corr.head(8).items():
            print(f"{feature:20s}: {value:+.3f}")


if __name__ == "__main__":
    main()
