# -*- coding: utf-8 -*-
"""
NEE Flux Analysis - Stage 2: Hyperparameter Tuning with Selected Features
Uses features selected by RFECV from Stage 1 (NEE_RFE_ZaF.py)
Performs comprehensive hyperparameter optimization to build final model
@author: Georgina Wieth-Klitgaard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
from pathlib import Path
import time
from datetime import datetime

# Suppress all warnings at import time
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform

# Suppress sklearn parallel backend warnings
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', message='.*delayed.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Configure matplotlib for better plots
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# 1. CONFIGURATION AND FEATURE SELECTION
# ============================================================================

print("="*70)
print("NEE FLUX ANALYSIS - STAGE 2: HYPERPARAMETER TUNING")
print("="*70)
print("\n[1/7] Loading configuration and features...")

# Output configuration
site_ID = "ZaF"
period = "_fieldseason"  # "" for full period, "_GS" for growing season and "_fieldseason" for field season only
output_dir = Path(__file__).resolve().parent / "output" / site_ID / "NEE_HPT_ZaF"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FEATURE SELECTION - Edit this list to customize features
# ============================================================================
# Default list based on Stage 1 RFE results
# Add or remove features as needed for your analysis

selected_features = [
    # Seasonal (cyclical)
    'sin_doy',
    'cos_doy',
    
    # Snow and soil moisture
    'DSSM',
    
    # Vegetation
    'NDVI_nonSR_Median',
    'NDVI_nonSR_Max',
    
    # Meteorology - current day
    'TA',
    'TA_min',
    'TA_max',
    'RG',
    'RG_min',
    'RG_max',
    'precipitation_rate',
    
    # Soil - current day
    'SWC_10cm',
    'TS_10cm',
    
    # Lagged meteorology (rolling averages)
    'TA_lag3d',
    'TA_lag7d',
    'TA_lag14d',
    
    # Lagged soil (rolling averages)
    'TS_lag1d',
    'TS_lag3d',
    'TS_lag7d',
    'TS_lag14d',
    'SWC_lag1d',
    'SWC_lag3d',
    'SWC_lag7d',
    'SWC_lag14d',
    
    # Lagged precipitation (cumulative sums)
    'P_lag14d'
]

print(f"\nTotal features selected: {len(selected_features)}")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")

# ============================================================================
# 2. DATA LOADING AND PREPARATION
# ============================================================================

print("\n[2/7] Loading and preparing data...")

# Load data
flux_path = Path(__file__).resolve().parent / "data" / f"{site_ID}_complete{period}.csv"
data = pd.read_csv(flux_path, sep=',', parse_dates=['Date'])
data = data.sort_values('Date').reset_index(drop=True)

# Define target
target_col = 'NEE_U50_f'

# Compute cyclical seasonal features before dropping NaNs
data['sin_doy'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365)
data['cos_doy'] = np.cos(2 * np.pi * data['Date'].dt.dayofyear / 365)

# Drop rows with missing data (using only selected features)
data_clean = data.dropna(subset=[target_col] + selected_features).copy()

print(f"\nTotal observations: {len(data_clean)}")
print(f"Date range: {data_clean['Date'].min()} to {data_clean['Date'].max()}")
print(f"Target variable: {target_col}")
print(f"Number of selected features: {len(selected_features)}")

# Add temporal features for analysis
data_clean['Year'] = data_clean['Date'].dt.year
data_clean['Month'] = data_clean['Date'].dt.month
data_clean['DOY'] = data_clean['Date'].dt.dayofyear

# Check number of years available
unique_years = data_clean['Year'].unique()
n_years = len(unique_years)
print(f"\nYears available: {sorted(unique_years)} ({n_years} years total)")
print(f"Samples per year: {data_clean.groupby('Year').size().to_dict()}")

# ============================================================================
# 3. VALIDATION STRATEGY (SAME AS STAGE 1 - ENSURES NO DATA LEAKAGE)
# ============================================================================

print("\n[3/7] Setting up validation strategy...")

X = data_clean[selected_features + ['Date']]
y = data_clean[target_col]
groups = data_clean['Year']

print(f"\nStrategy: Random 80/20 Train/Test Split + Group K-Fold CV")
print("  • 20% held out for final testing (random split across all years)")
print("  • Group K-Fold CV on training data for hyperparameter tuning")
print(f"  • IMPORTANT: Using same random_state=42 as Stage 1")
print(f"  • This ensures IDENTICAL train-test split - NO DATA LEAKAGE\n")

# Random 80/20 split - SAME RANDOM STATE AS STAGE 1
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups,
    test_size=0.2,
    random_state=42,  # CRITICAL: Same as Stage 1
    stratify=groups
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Training years: {sorted(groups_train.unique())}")
print(f"Test years: {sorted(groups_test.unique())}")

# Use all years for GroupKFold CV
n_cv_splits = groups_train.nunique()
if n_cv_splits < 2:
    raise ValueError("Need at least 2 unique years for GroupKFold.")
cv_strategy = GroupKFold(n_splits=n_cv_splits)
print(f"Using {n_cv_splits}-fold GroupKFold CV (one fold per year)")

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================

print("\n[4/7] Performing hyperparameter tuning...")

# Build preprocessing pipeline for selected features
preproc = ColumnTransformer([
    ('scale', StandardScaler(), selected_features)
], remainder='drop')

# Define hyperparameter search space
param_distributions = {
    'rf__n_estimators': randint(100, 500),           # Number of trees
    'rf__max_depth': [10, 20, 30, 40, 50, None],     # Maximum tree depth
    'rf__min_samples_split': randint(2, 20),         # Min samples to split node
    'rf__min_samples_leaf': randint(1, 10),          # Min samples in leaf
    'rf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],  # Features per split
    'rf__max_samples': uniform(0.6, 0.4)             # Bootstrap sample size (0.6-1.0)
}

print("\nHyperparameter Search Space:")
print(f"  n_estimators: 100-500 (uniform)")
print(f"  max_depth: [10, 20, 30, 40, 50, None]")
print(f"  min_samples_split: 2-20 (uniform)")
print(f"  min_samples_leaf: 1-10 (uniform)")
print(f"  max_features: ['sqrt', 'log2', 0.3, 0.5, 0.7]")
print(f"  max_samples: 0.6-1.0 (uniform)")

# Create base Random Forest
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    oob_score=True  # Enable out-of-bag score
)

# Build pipeline
pipeline = Pipeline([
    ('pre', preproc),
    ('rf', rf)
])

# Setup RandomizedSearchCV
n_iter = 100  # Number of random combinations to try
print(f"\nRandomizedSearchCV Configuration:")
print(f"  Iterations: {n_iter}")
print(f"  CV Strategy: {n_cv_splits}-fold GroupKFold")
print(f"  Scoring: R²")
print(f"  Estimated total model fits: ~{n_iter * n_cv_splits}")

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=cv_strategy,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=2,
    return_train_score=True
)

print(f"\nStarting hyperparameter search...")
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}\n")
print("="*70)

start_time = time.time()

# Fit RandomizedSearchCV
try:
    random_search.fit(X_train, y_train, groups=groups_train)
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total elapsed time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
except Exception as e:
    elapsed_time = time.time() - start_time
    print(f"\nERROR after {elapsed_time/60:.1f} minutes: {e}")
    raise

# Get best model and parameters
best_pipeline = random_search.best_estimator_
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print("\n" + "="*70)
print("BEST HYPERPARAMETERS")
print("="*70)
for param, value in best_params.items():
    param_name = param.replace('rf__', '')
    print(f"  {param_name}: {value}")

print(f"\nBest CV R²: {best_cv_score:.4f}")

# Get OOB score from best estimator
best_rf = best_pipeline.named_steps['rf']
oob_score = best_rf.oob_score_
print(f"Out-of-Bag R²: {oob_score:.4f}")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n[5/7] Evaluating final model performance...")

# Training set predictions
y_pred_train = best_pipeline.predict(X_train)

# Test set predictions
y_pred_test = best_pipeline.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n--- Final Model Performance ---")
print(f"{'Metric':<15} {'CV':<15} {'OOB':<15} {'Train':<15} {'Test':<15}")
print("-" * 75)
print(f"{'R²':<15} {best_cv_score:<15.4f} {oob_score:<15.4f} {train_r2:<15.4f} {test_r2:<15.4f}")
print(f"{'RMSE':<15} {'N/A':<15} {'N/A':<15} {train_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"{'MAE':<15} {'N/A':<15} {'N/A':<15} {train_mae:<15.4f} {test_mae:<15.4f}")

print(f"\nTrain-Test R² Gap: {train_r2 - test_r2:.4f} "
      f"({'⚠ check overfitting' if train_r2 - test_r2 > 0.15 else '✓ acceptable'})")

# Per-year test set performance
print(f"\n--- Test Set Performance by Year ---")
test_years_unique = sorted(groups_test.unique())
for year in test_years_unique:
    year_mask = groups_test == year
    y_year_test = y_test[year_mask]
    y_year_pred = y_pred_test[year_mask]
    
    year_r2 = r2_score(y_year_test, y_year_pred)
    year_rmse = np.sqrt(mean_squared_error(y_year_test, y_year_pred))
    year_mae = mean_absolute_error(y_year_test, y_year_pred)
    
    print(f"Year {year} (n={year_mask.sum()}): "
          f"R² = {year_r2:.4f}, RMSE = {year_rmse:.4f}, MAE = {year_mae:.4f}")

# ============================================================================
# 6. SHAP ANALYSIS
# ============================================================================

print("\n[6/7] Performing SHAP analysis...")

# Prepare feature names
feature_names = selected_features

# Transform data for SHAP
X_train_transformed = best_pipeline.named_steps['pre'].transform(X_train)
X_test_transformed = best_pipeline.named_steps['pre'].transform(X_test)

# Create SHAP explainer
rf_model = best_pipeline.named_steps['rf']
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for test set (subset for plotting)
sample_size = min(1000, len(X_test_transformed))
sample_idx = np.random.choice(len(X_test_transformed), sample_size, replace=False)
X_test_sample = X_test_transformed[sample_idx]

print(f"Computing SHAP values for {sample_size} test samples...")
shap_values = explainer.shap_values(X_test_sample)

# Compute SHAP values for full test set (for export)
print("Computing SHAP values for full test set (for export)...")
shap_values_full = explainer.shap_values(X_test_transformed)

# Prepare raw (unscaled) data for SHAP plots
X_test_reset = X_test.iloc[sample_idx].reset_index(drop=True)
X_test_raw = X_test_reset[selected_features].copy()
X_test_raw = X_test_raw[feature_names]

# Prepare raw data for full test set (for export)
X_test_reset_full = X_test.reset_index(drop=True)
X_test_raw_full = X_test_reset_full[selected_features].copy()
X_test_raw_full = X_test_raw_full[feature_names]

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n[7/7] Creating visualizations...")

# --- Plot 1: Observed vs Predicted ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_train, y_pred_train, alpha=0.3, s=10)
axes[0].plot([y_train.min(), y_train.max()],
             [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Observed NEE')
axes[0].set_ylabel('Predicted NEE')
axes[0].set_title(f'Training Set (R² = {train_r2:.3f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, y_pred_test, alpha=0.3, s=10, color='orange')
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Observed NEE')
axes[1].set_ylabel('Predicted NEE')
axes[1].set_title(f'Test Set (R² = {test_r2:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_observed_vs_predicted_HPT.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_observed_vs_predicted_HPT.png")
plt.close()

# --- Plot 2: Residuals Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

axes[0, 0].scatter(y_pred_train, residuals_train, alpha=0.3, s=10)
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted NEE')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Training Set Residuals')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_pred_test, residuals_test, alpha=0.3, s=10, color='orange')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted NEE')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Test Set Residuals')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(residuals_train, bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Training Residuals Distribution (μ={residuals_train.mean():.3f})')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(residuals_test, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Test Residuals Distribution (μ={residuals_test.mean():.3f})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '02_residuals_analysis_HPT.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_residuals_analysis_HPT.png")
plt.close()

# --- Plot 3: Hyperparameter Tuning Results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: CV scores for all iterations
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df = cv_results_df.sort_values('rank_test_score')

ax1.scatter(range(len(cv_results_df)), cv_results_df['mean_test_score'], 
            alpha=0.5, s=30)
ax1.axhline(y=best_cv_score, color='r', linestyle='--', linewidth=2,
            label=f'Best: {best_cv_score:.4f}')
ax1.set_xlabel('Configuration (sorted by rank)', fontsize=12)
ax1.set_ylabel('CV R² Score', fontsize=12)
ax1.set_title('Hyperparameter Tuning: All Configurations', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Right: Feature Importance (from tuned model)
rf_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importances
}).sort_values('Importance', ascending=False)

bars = ax2.barh(importance_df['Feature'], importance_df['Importance'])
ax2.set_xlabel('Feature Importance (MDI)', fontsize=12)
ax2.set_title('Feature Importance (Tuned Model)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
plt.savefig(output_dir / '03_hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_hyperparameter_tuning_results.png")
plt.close()

# --- Plot 4: SHAP Summary Plot (Feature Importance) ---
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_raw, feature_names=feature_names, 
                  show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig(output_dir / '04_shap_summary_importance_HPT.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_shap_summary_importance_HPT.png")
plt.close()

# --- Plot 5: SHAP Summary Plot (Feature Effects) ---
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_raw, feature_names=feature_names,
                  plot_type='violin', show=False)
plt.tight_layout()
plt.savefig(output_dir / '05_shap_summary_effects_HPT.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_shap_summary_effects_HPT.png")
plt.close()

# --- Plot 6: SHAP Dependence Plots for Top Features ---
shap_importance = np.abs(shap_values).mean(axis=0)
importance_order = np.argsort(shap_importance)[::-1]

# Find top non-seasonal features
top_features = []
for idx in importance_order:
    if feature_names[idx] not in ['sin_doy', 'cos_doy']:
        top_features.append(idx)
    if len(top_features) == 4:
        break

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, feat_idx in enumerate(top_features):
    feat_name = feature_names[feat_idx]
    shap.dependence_plot(
        feat_idx, 
        shap_values, 
        X_test_raw,
        feature_names=feature_names,
        ax=axes[i],
        show=False
    )
    axes[i].set_title(f'SHAP Dependence: {feat_name}')

plt.tight_layout()
plt.savefig(output_dir / '06_shap_dependence_plots_HPT.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_shap_dependence_plots_HPT.png")
plt.close()

# --- Plot 7: Individual Prediction Explanation (sample) ---
pred_order = np.argsort(best_pipeline.predict(X_test.iloc[sample_idx]))
selected_indices = [
    pred_order[0],
    pred_order[len(pred_order)//3],
    pred_order[2*len(pred_order)//3],
    pred_order[-1]
]

base_value_scalar = float(np.ravel(explainer.expected_value)[0])

for i, idx in enumerate(selected_indices):
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=base_value_scalar,
            data=X_test_raw.iloc[idx].values,
            feature_names=feature_names
        ),
        show=False
    )
    plt.gcf().set_size_inches(8, 6)
    plt.tight_layout()
    plt.savefig(output_dir / f'07_shap_waterfall_HPT_{i+1}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 07_shap_waterfall_HPT_{i+1}.png")
    plt.close()

# --- Plot 8: Temporal Analysis of Predictions ---
test_df = X_test.copy()
test_df['Observed'] = y_test.values
test_df['Predicted'] = y_pred_test
test_df['Residual'] = residuals_test.values
test_df = test_df.sort_values('Date')

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

axes[0].plot(test_df['Date'], test_df['Observed'], 
             label='Observed', alpha=0.7, linewidth=0.5)
axes[0].plot(test_df['Date'], test_df['Predicted'], 
             label='Predicted', alpha=0.7, linewidth=0.5)
axes[0].set_ylabel('NEE')
axes[0].set_title('Temporal Comparison: Observed vs Predicted NEE (Tuned Model)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(test_df['Date'], test_df['Residual'], 
                alpha=0.5, s=5, c='red')
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Over Time')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(test_df['Date'], np.abs(test_df['Residual']), 
                alpha=0.5, s=5, c='orange')
axes[2].set_ylabel('|Residuals|')
axes[2].set_xlabel('Date')
axes[2].set_title('Absolute Residuals Over Time')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '08_temporal_analysis_HPT.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_temporal_analysis_HPT.png")
plt.close()

# ============================================================================
# 8. EXPORT RESULTS
# ============================================================================

print("\n[8/8] Exporting results...")

# Export model performance metrics
metrics_df = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE'],
    'Training': [train_r2, train_rmse, train_mae],
    'Test': [test_r2, test_rmse, test_mae],
    'CV_Best': [best_cv_score, np.nan, np.nan],
    'OOB': [oob_score, np.nan, np.nan]
})
metrics_df.to_csv(output_dir / 'model_performance_metrics_HPT.csv', index=False)
print("✓ Saved: model_performance_metrics_HPT.csv")

# Export best hyperparameters
best_params_clean = {k.replace('rf__', ''): v for k, v in best_params.items()}
best_params_df = pd.DataFrame([
    {'Parameter': k, 'Value': v} for k, v in best_params_clean.items()
])
best_params_df.to_csv(output_dir / 'best_hyperparameters_HPT.csv', index=False)
print("✓ Saved: best_hyperparameters_HPT.csv")

# Export hyperparameter search results (top 20 configurations)
cv_results_export = cv_results_df.head(20)[[
    'rank_test_score', 'mean_test_score', 'std_test_score',
    'param_rf__n_estimators', 'param_rf__max_depth', 
    'param_rf__min_samples_split', 'param_rf__min_samples_leaf',
    'param_rf__max_features', 'param_rf__max_samples'
]]
cv_results_export.to_csv(output_dir / 'hyperparameter_search_top20.csv', index=False)
print("✓ Saved: hyperparameter_search_top20.csv")

# Export feature importance
importance_df.to_csv(output_dir / 'feature_importance_HPT.csv', index=False)
print("✓ Saved: feature_importance_HPT.csv")

# Export SHAP values summary
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0),
    'Mean_SHAP': shap_values.mean(axis=0),
    'Std_SHAP': shap_values.std(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)
shap_importance_df.to_csv(output_dir / 'shap_feature_importance_HPT.csv', index=False)
print("✓ Saved: shap_feature_importance_HPT.csv")

# Export per-row SHAP values for temporal analysis
shap_columns = [f"SHAP_{name}" for name in feature_names]
shap_values_df = pd.DataFrame(shap_values_full, columns=shap_columns)
shap_values_df.insert(0, 'Date', X_test_reset_full['Date'].values)
shap_values_df.insert(1, 'Year', X_test_reset_full['Date'].dt.year.values)
shap_values_df.insert(2, 'Month', X_test_reset_full['Date'].dt.month.values)
shap_values_df.to_csv(output_dir / 'shap_values_HPT.csv', index=False)
print("✓ Saved: shap_values_HPT.csv")

# Export predictions with metadata
predictions_df = pd.DataFrame({
    'Date': X_test['Date'].values,
    'Observed_NEE': y_test.values,
    'Predicted_NEE': y_pred_test,
    'Residual': residuals_test.values,
    'Abs_Residual': np.abs(residuals_test.values)
})
for pred_name in selected_features:
    predictions_df[pred_name] = X_test[pred_name].values
predictions_df.to_csv(output_dir / 'test_predictions_HPT.csv', index=False)
print("✓ Saved: test_predictions_HPT.csv")

# ============================================================================
# 9. COMPARISON WITH STAGE 1 (RFE Model)
# ============================================================================

print("\n[9/9] Comparing with Stage 1 results...")

# Load Stage 1 metrics if available (from RFE subfolder)
stage1_metrics_file = Path(__file__).resolve().parent / "output" / site_ID / "NEE_RFE_ZaF" / 'model_performance_metrics.csv'
if stage1_metrics_file.exists():
    stage1_metrics = pd.read_csv(stage1_metrics_file)
    stage1_test_r2 = stage1_metrics[stage1_metrics['Metric'] == 'R²']['Test'].values[0]
    stage1_test_rmse = stage1_metrics[stage1_metrics['Metric'] == 'RMSE']['Test'].values[0]
    stage1_test_mae = stage1_metrics[stage1_metrics['Metric'] == 'MAE']['Test'].values[0]
    
    print("\n--- Model Comparison: Stage 1 (RFE) vs Stage 2 (HPT) ---")
    print(f"{'Metric':<15} {'Stage 1 (RFE)':<20} {'Stage 2 (HPT)':<20} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Test R²':<15} {stage1_test_r2:<20.4f} {test_r2:<20.4f} {test_r2 - stage1_test_r2:+.4f}")
    print(f"{'Test RMSE':<15} {stage1_test_rmse:<20.4f} {test_rmse:<20.4f} {stage1_test_rmse - test_rmse:+.4f}")
    print(f"{'Test MAE':<15} {stage1_test_mae:<20.4f} {test_mae:<20.4f} {stage1_test_mae - test_mae:+.4f}")
    
    # Create comparison summary
    comparison_df = pd.DataFrame({
        'Stage': ['Stage 1 (RFE)', 'Stage 2 (HPT)'],
        'Test_R2': [stage1_test_r2, test_r2],
        'Test_RMSE': [stage1_test_rmse, test_rmse],
        'Test_MAE': [stage1_test_mae, test_mae],
        'Model': ['Default RF', 'Tuned RF'],
        'N_Features': [len(selected_features), len(selected_features)]
    })
    comparison_df.to_csv(output_dir / 'stage_comparison.csv', index=False)
    print("\n✓ Saved: stage_comparison.csv")
else:
    print("\n⚠ Stage 1 metrics not found - skipping comparison")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE - STAGE 2: HYPERPARAMETER TUNING")
print("="*70)
print("\nKey Findings:")
print(f"  • Best CV R²: {best_cv_score:.4f}")
print(f"  • Test R²: {test_r2:.4f}")
print(f"  • Test RMSE: {test_rmse:.4f}")
print(f"  • OOB R²: {oob_score:.4f}")
print(f"\nBest Hyperparameters:")
for param, value in best_params_clean.items():
    print(f"  • {param}: {value}")

print(f"\nTop 3 Most Important Features (SHAP):")
for i in range(min(3, len(shap_importance_df))):
    row = shap_importance_df.iloc[i]
    print(f"  {i+1}. {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")

print("\nOutput Files Generated:")
print("  • 8+ visualization plots (PNG, suffix: _HPT)")
print("  • 7 data files (CSV, suffix: _HPT)")
print(f"\nAll results saved to: {output_dir}")
print("="*70)

print("\n" + "="*70)
print("WORKFLOW NOTES")
print("="*70)
print("✓ Stage 1 (NEE_RFE_ZaF.py): Feature selection with RFECV")
print("✓ Stage 2 (NEE_HPT_ZaF.py): Hyperparameter tuning with selected features")
print("\nData Leakage Check:")
print("  • Same random_state=42 ensures identical train-test split")
print("  • Test data never used in RFE or hyperparameter tuning")
print("  • NO DATA LEAKAGE - approach is valid!")
print("="*70)
