# GHI Forecasting Project

This project builds, benchmarks, and compares a comprehensive suite of solar irradiance forecasting models for an NSRDB weather site. What began as a baseline LSTM forecasting pipeline has evolved into a rigorous framework evaluating 9 distinct architectures—ranging from classical machine learning to deep hybrid networks—culminating in a state-of-the-art Residual Hybrid model.

The primary goal is to forecast hourly Global Horizontal Irradiance (GHI) and produce robust, publication-ready evaluations spanning multi-day horizons, sub-daily regimes (clear sky, cloudy, partly cloudy), and specific feature ablation tests.

## What This Project Does

The repository ingests hourly NSRDB CSV files, engineers sophisticated time-series and solar-geometry features, trains a diverse array of models, and rigorously evaluates them on a chronological holdout split. 

Key capabilities include:
1. **Comprehensive Benchmarking:** Training and evaluating 9 different models (ANN, DNN, LSTM, CNN-DNN, CNN-LSTM, CNN-A-LSTM, GBT, SVM, and a Hybrid Residual model).
2. **Advanced Pipeline:** Handling data parsing, strict temporal splitting (preventing data leakage), sequence generation, and robust scaling.
3. **Publication-Grade Visualizations:** Automatically generating 11 distinct, high-quality figures (heatmaps, temporal tracking, residual distributions, regime breakdowns) for research dissemination.
4. **Regime-Aware Metrics:** Breaking down model performance by clear sky, partly cloudy, and cloudy conditions.

## Dataset

The dataset folder contains hourly NSRDB CSV files for a single site in California:
- **Location ID:** 820809
- **Coordinates:** 34.12° N, -118.39° W
- **Time zone:** UTC-8
- **Years included:** 2018 to 2021

Features span standard meteorology (Temperature, Relative Humidity, Wind Speed, Pressure) alongside derived solar geometries (Zenith Angle, Cosine Zenith) and temporal identifiers.

## Technologies Used

- **Python** (Core Language)
- **TensorFlow / Keras** (Deep Learning: ANN, DNN, LSTM, CNN variants)
- **LightGBM & Scikit-Learn** (Classical ML: GBT, SVM, Residual matching and regime splitting)
- **Pandas & NumPy** (Data manipulation, sequence building)
- **Matplotlib & Seaborn** (High-quality figure generation)

## Project Structure

``text
code/
  benchmark/                      Core benchmarking pipeline
    models/                       Definitions for all 9 architectures
    data_loader.py                NSRDB ingestion and cleaning
    features.py                   Feature engineering
    run.py                        Main benchmark execution script
    ...                           (Validation, Tuning, Ablation, Leakage Audits)
  dataset/                        NSRDB CSV files
  models/                         Legacy implementations & automated agents
    agent/                        Sprint automation (ghi_improvement_agent, tier2)
    baseline_lstm/                Original baseline LSTM 
    attention_lstm/               Original attention LSTM
    residual_hybrid/              Original residual hybrid
  outputs/
    benchmark/                    Saved artifacts, logs, metrics, predictions
    figures/                      11 publication-ready figures & figure_report.md
    plots/                        Organized legacy exploratory plots
``

## Evaluated Models

This project evaluates the following predictors:

1. **Classical ML:** GBT (LightGBM), SVM (SVR-RBF)
2. **Standard Deep Learning:** ANN, DNN, LSTM
3. **Spatiotemporal/Convolutional:** CNN-DNN, CNN-LSTM, CNN-A-LSTM (Attention)
4. **Hybrid:** Hybrid Residual Model (A sequential model coupled with an error-correcting gradient-boosted calibrator).

*Note: The **Hybrid Residual** consistently outperforms the other structures by shifting the focus from re-learning global irradiance to correcting baseline systematic errors, showing massive drops in MAE and RMSE.*

## Execution and Capabilities

### 1. Run the Full Benchmark Suite
Execute the benchmarking pipeline to train all models and generate predictions, logs, and robust metric profiles:
``powershell
python benchmark/run.py
``

### 2. Generate Publication Figures
Generate the complete set of 11 publication-grade charts (incorporating the 9 models):
``powershell
python outputs/figures/generate_publication_figures.py
``
*Generates outputs in outputs/figures/ and updates outputs/figures/figure_report.md.*

### 3. Legacy Automated Agents
For exploring iterative architectural improvements via LLM-based autonomous agents:
``powershell
python models/agent/tier2_automation.py
``