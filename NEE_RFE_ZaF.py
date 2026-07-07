# -*- coding: utf-8 -*-
"""
NEE Flux Analysis with SHAP Interpretation
Focus: Model interpretation, predictor importance, and environmental relationships
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

# Suppress sklearn parallel backend warnings
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', message='.*delayed.*')
# Suppress numpy correlation warnings when computing sensitivity
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Configure matplotlib for better plots
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

print("="*70)
print("NEE FLUX ANALYSIS WITH SHAP INTERPRETATION")
print("="*70)
print("\n[1/6] Loading and preparing data...")

# Output configuration
site_ID = "ZaF"
period = "_fieldseason" # " " for full period, "_fieldseason" for field season only 
output_dir = Path(__file__).resolve().parent / "output" / site_ID / "NEE_RFE_ZaF"
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
flux_path = Path(__file__).resolve().parent / "data" / f"{site_ID}_complete{period}.csv"
data = pd.read_csv(flux_path, sep=',', parse_dates=['Date'])
data = data.sort_values('Date').reset_index(drop=True)

"""
# Feature engineering
data['T_lag1'] = data['TS_0.02'].shift(1)
"""

# Define target and predictors
target_col = 'NEE_U50_f'
predictors = [
    # Seasonal (cyclical) - subject to RFECV
    'sin_doy',
    'cos_doy',
    
    # Environmental conditions
    'Snow_Cover_Percentage',
    'DSSM',
    'D_SNOW',
    
    # Vegetation
   # 'NDVI_nonSR_Min',
    #'NDVI_nonSR_Median',
    'NDVI_nonSR_Max',
    
    # Meteorology - current day
    'RH',
    'TA',
    'TA_min',
    'TA_max',
    'RG',
    'RG_min',
    'RG_max',
    'VPD_f',
    'precipitation_rate',
    
    # Soil - current day
    'SWC_10cm',
    'TS_10cm',
    
    # Lagged meteorology (rolling averages)
    'TA_lag1d',
    'TA_lag3d',
    'TA_lag7d',
    'TA_lag14d',
    
    # Lagged soil (rolling averages)
    #'TS_lag1d',
    'TS_lag3d',
    'TS_lag7d',
    'TS_lag14d',
    'SWC_lag1d',
    'SWC_lag3d',
    'SWC_lag7d',
    'SWC_lag14d',
    
    # Lagged precipitation (cumulative sums)
    'P_lag1d',
    'P_lag3d',
    'P_lag7d',
    'P_lag14d'
]



# Compute cyclical seasonal features before dropping NaNs
data['sin_doy'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365)
data['cos_doy'] = np.cos(2 * np.pi * data['Date'].dt.dayofyear / 365)

# Drop rows with missing data
data_clean = data.dropna(subset=[target_col] + predictors).copy()

print(f"Total observations: {len(data_clean)}")
print(f"Date range: {data_clean['Date'].min()} to {data_clean['Date'].max()}")
print(f"Target variable: {target_col}")
print(f"Predictors: {', '.join(predictors)}")

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
# 2. VALIDATION STRATEGY
# ============================================================================

print("\n[2/6] Setting up validation strategy...")

X = data_clean[predictors + ['Date']]
y = data_clean[target_col]
groups = data_clean['Year']

print(f"\nStrategy: Random 80/20 Train/Test Split + Group K-Fold CV")
print("  • 20% held out for final testing (random split across all years)")
print("  • Group K-Fold CV on training data\n")

# Random 80/20 split
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups,
    test_size=0.2,
    random_state=42,
    stratify=groups
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Training years: {sorted(groups_train.unique())}")
print(f"Test years: {sorted(groups_test.unique())}")

# Use all 8 years for GroupKFold CV (one fold per year)
n_cv_splits = groups_train.nunique()
if n_cv_splits < 2:
    raise ValueError("Need at least 2 unique years for GroupKFold.")
cv_strategy = GroupKFold(n_splits=n_cv_splits)
print(f"Using {n_cv_splits}-fold GroupKFold CV (one fold per year)")

# ============================================================================
# 3. RECURSIVE FEATURE ELIMINATION WITH CROSS-VALIDATION (RFECV)
# ============================================================================

print("\n[3/6] Performing feature selection with RFECV...")

# Build preprocessing pipeline (scale all predictors including sin_doy and cos_doy)
preproc = ColumnTransformer([
    ('scale', StandardScaler(), predictors)
], remainder='drop')

# Create Random Forest with DEFAULT parameters (no hyperparameter tuning yet)
rf_default = RandomForestRegressor(
    n_estimators=200,      # Default
    max_depth=8,           # Default (unlimited)
    max_features='log2',   # log2(n_features)
    random_state=42,
    min_samples_leaf=2,
    min_samples_split=5,
    n_jobs=-1
)

print(f"\nRandom Forest Configuration (Default Parameters):")
print(f"  n_estimators: 100 (default)")
print(f"  max_depth: 10")
print(f"  min_samples_split: 10")
print(f"  min_samples_leaf: 2")
print(f"  max_features: 'log2' (log2 of n_features)")
print(f"\nFeature Selection Strategy:")
print(f"  Method: RFECV (Recursive Feature Elimination with CV)")
print(f"  CV Strategy: {n_cv_splits}-fold GroupKFold (one fold per year)")
print(f"  Scoring: R²")
print(f"  Step size: 1 feature removed per iteration")
print(f"\n" + "="*70)
print("STARTING FEATURE ELIMINATION")
print("="*70)
print(f"Starting with {len(predictors)} features")
print(f"This will test models with {len(predictors)} down to 1 feature")
print(f"Each feature count tested with {n_cv_splits}-fold CV")
print(f"Estimated total model fits: ~{len(predictors) * n_cv_splits}")
print(f"Progress updates will show after each feature elimination step\n")

# Create RFECV selector with maximum verbosity
selector = RFECV(
    estimator=rf_default,
    step=1,                # Remove 1 feature at a time (most thorough)
    cv=cv_strategy,
    scoring='r2',
    n_jobs=-1,
    verbose=2              # Maximum verbosity for detailed progress
)

# Build full pipeline with preprocessing + feature selection
feature_selection_pipeline = Pipeline([
    ('pre', preproc),
    ('selector', selector)
])

print(f"Pipeline created. Beginning feature elimination...")
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}\n")

start_time = time.time()

# Fit RFECV with progress tracking
try:
    feature_selection_pipeline.fit(X_train, y_train, selector__groups=groups_train)
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"RFECV COMPLETE")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total elapsed time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
except Exception as e:
    elapsed_time = time.time() - start_time
    print(f"\nERROR after {elapsed_time/60:.1f} minutes: {e}")
    raise

# Get results
n_features_optimal = selector.n_features_
feature_support = selector.support_
feature_ranking = selector.ranking_

# Extract selected and removed features (all predictors evaluated equally)
selected_features = [predictors[i] for i in range(len(predictors)) if feature_support[i]]
removed_features = [predictors[i] for i in range(len(predictors)) if not feature_support[i]]

print("\n" + "="*70)
print("FEATURE SELECTION RESULTS")
print("="*70)
print(f"\nOptimal number of features: {n_features_optimal}/{len(predictors)}")
print(f"Features removed: {len(removed_features)}")

if removed_features:
    print(f"\n--- Removed Features ---")
    for feat in removed_features:
        rank = feature_ranking[predictors.index(feat)]
        print(f"  {feat} (rank: {rank})")

print(f"\n--- Selected Features ---")
for feat in selected_features:
    print(f"  ✓ {feat}")

print(f"\n--- CV Performance vs Number of Features ---")
for n_feat, score in enumerate(selector.cv_results_['mean_test_score'], start=1):
    marker = " ← OPTIMAL" if n_feat == n_features_optimal else ""
    print(f"  {n_feat:2d} features: R² = {score:.4f}{marker}")

# Prepare data with selected features only (keep Date for later export/plotting)
X_train_selected = X_train[selected_features + ['Date']]
X_test_selected = X_test[selected_features + ['Date']]

# Build NEW preprocessing pipeline for selected features only
preproc_selected = ColumnTransformer([
    ('scale', StandardScaler(), selected_features)
], remainder='drop')

# Build final model with selected features
final_pipeline = Pipeline([
    ('pre', preproc_selected),
    ('rf', rf_default)
])

# Fit on training data with selected features
final_pipeline.fit(X_train_selected, y_train)
best_pipeline = final_pipeline  # For compatibility with downstream code
# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================

print("\n[4/6] Evaluating model performance with selected features...")

# Use RFECV CV results (already computed)
cv_scores = selector.cv_results_['mean_test_score']
cv_best_score = cv_scores[n_features_optimal - 1]  # -1 for 0-indexing

print(f"\nBest CV R² (with {n_features_optimal} features): {cv_best_score:.4f}")

# Test set evaluation with selected features
y_pred_train = best_pipeline.predict(X_train_selected)
y_pred_test = best_pipeline.predict(X_test_selected)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n--- Model Performance (with selected features) ---")
print(f"{'Metric':<15} {'CV':<15} {'Train':<15} {'Test':<15}")
print("-" * 60)
print(f"{'R²':<15} {cv_best_score:<15.4f} {train_r2:<15.4f} {test_r2:<15.4f}")
print(f"{'RMSE':<15} {'N/A':<15} {train_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"{'MAE':<15} {'N/A':<15} {train_mae:<15.4f} {test_mae:<15.4f}")

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

year_metrics = None

# ============================================================================
# 5. SHAP ANALYSIS - PREDICTOR IMPORTANCE AND RELATIONSHIPS
# ============================================================================

print("\n[5/6] Performing SHAP analysis...")

# Prepare feature names (after transformation) - use selected features only
feature_names = selected_features

# Transform data for SHAP - use selected feature datasets
X_train_transformed = best_pipeline.named_steps['pre'].transform(X_train_selected)
X_test_transformed = best_pipeline.named_steps['pre'].transform(X_test_selected)

# Create SHAP explainer
rf_model = best_pipeline.named_steps['rf']
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for test set (subset for efficiency)
sample_size = min(1000, len(X_test_transformed))
sample_idx = np.random.choice(len(X_test_transformed), sample_size, replace=False)
X_test_sample = X_test_transformed[sample_idx]

print(f"Computing SHAP values for {sample_size} test samples...")
shap_values = explainer.shap_values(X_test_sample)

# Compute SHAP values for full test set to enable temporal aggregation
print("Computing SHAP values for full test set (for export)...")
shap_values_full = explainer.shap_values(X_test_transformed)

# Prepare raw (unscaled) data for SHAP plots - use selected features
X_test_reset = X_test_selected.iloc[sample_idx].reset_index(drop=True)
X_test_raw = X_test_reset[selected_features].copy()
X_test_raw = X_test_raw[feature_names]

# Prepare raw (unscaled) data for full test set (for export) - use selected features
X_test_reset_full = X_test_selected.reset_index(drop=True)
X_test_raw_full = X_test_reset_full[selected_features].copy()
X_test_raw_full = X_test_raw_full[feature_names]

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n[6/6] Creating visualizations...")

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
plt.savefig(output_dir / '01_observed_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_observed_vs_predicted.png")
plt.close()

# --- Plot 2: Residuals Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

# Residuals vs Predicted
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

# Residual distributions
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
plt.savefig(output_dir / '02_residuals_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_residuals_analysis.png")
plt.close()

# --- Plot 3: Feature Selection Results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: CV R² vs Number of Features
n_features_range = range(1, len(selector.cv_results_['mean_test_score']) + 1)
ax1.plot(n_features_range, selector.cv_results_['mean_test_score'], 'o-', linewidth=2, markersize=6)
ax1.axvline(x=n_features_optimal, color='r', linestyle='--', linewidth=2, 
            label=f'Optimal: {n_features_optimal} features')
ax1.set_xlabel('Number of Features', fontsize=12)
ax1.set_ylabel('CV R² Score', fontsize=12)
ax1.set_title('RFECV: Model Performance vs Feature Count', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Right: Feature Importance (from final RF model)
rf_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importances
}).sort_values('Importance', ascending=False)

bars = ax2.barh(importance_df['Feature'], importance_df['Importance'])
ax2.set_xlabel('Feature Importance (MDI)', fontsize=12)
ax2.set_title('Selected Features - RF Importance', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Color bars
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
plt.savefig(output_dir / '03_feature_selection_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_feature_selection_results.png")
plt.close()

# --- Plot 4: SHAP Summary Plot (Feature Importance) ---
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_raw, feature_names=feature_names, 
                  show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig(output_dir / '04_shap_summary_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_shap_summary_importance.png")
plt.close()

# --- Plot 5: SHAP Summary Plot (Feature Effects) ---
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_raw, feature_names=feature_names,
                  plot_type='violin', show=False)
plt.tight_layout()
plt.savefig(output_dir / '05_shap_summary_effects.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_shap_summary_effects.png")
plt.close()

# --- Plot 6: SHAP Dependence Plots for Top Features ---
# Get top 4 most important features (excluding sin/cos for clarity)
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
plt.savefig(output_dir / '06_shap_dependence_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_shap_dependence_plots.png")
plt.close()

# --- Plot 7: Individual Prediction Explanation (sample) ---
# Show SHAP values for a few individual predictions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# Select 4 interesting predictions (high/low predicted values)
pred_order = np.argsort(best_pipeline.predict(X_test.iloc[sample_idx]))
selected_indices = [
    pred_order[0],           # Lowest prediction
    pred_order[len(pred_order)//3],  # Lower third
    pred_order[2*len(pred_order)//3],  # Upper third
    pred_order[-1]           # Highest prediction
]

# SHAP waterfall expects a scalar base value for one prediction.
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
    plt.savefig(output_dir / f'07_shap_waterfall_{i+1}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 07_shap_waterfall_{i+1}.png")
    plt.close()

# --- Plot 8: Temporal Analysis of Predictions ---
test_df = X_test.copy()
test_df['Observed'] = y_test.values
test_df['Predicted'] = y_pred_test
test_df['Residual'] = residuals_test.values
test_df = test_df.sort_values('Date')

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Time series
axes[0].plot(test_df['Date'], test_df['Observed'], 
             label='Observed', alpha=0.7, linewidth=0.5)
axes[0].plot(test_df['Date'], test_df['Predicted'], 
             label='Predicted', alpha=0.7, linewidth=0.5)
axes[0].set_ylabel('NEE')
axes[0].set_title('Temporal Comparison: Observed vs Predicted NEE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals over time
axes[1].scatter(test_df['Date'], test_df['Residual'], 
                alpha=0.5, s=5, c='red')
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Over Time')
axes[1].grid(True, alpha=0.3)

# Absolute residuals (to check for patterns)
axes[2].scatter(test_df['Date'], np.abs(test_df['Residual']), 
                alpha=0.5, s=5, c='orange')
axes[2].set_ylabel('|Residuals|')
axes[2].set_xlabel('Date')
axes[2].set_title('Absolute Residuals Over Time')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '08_temporal_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_temporal_analysis.png")
plt.close()

# ============================================================================
# 7. EXPORT RESULTS
# ============================================================================

print("\n[7/7] Exporting results...")

# Export model performance metrics
metrics_df = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE'],
    'Training': [train_r2, train_rmse, train_mae],
    'Test': [test_r2, test_rmse, test_mae],
    'CV_Best': [cv_best_score, np.nan, np.nan],
    'N_Features': [n_features_optimal, np.nan, np.nan]
})
metrics_df.to_csv(output_dir / 'model_performance_metrics.csv', index=False)
print("✓ Saved: model_performance_metrics.csv")

# Export feature selection results
feature_selection_df = pd.DataFrame({
    'Feature': predictors,
    'Selected': feature_support,
    'Ranking': feature_ranking
}).sort_values('Ranking')
feature_selection_df.to_csv(output_dir / 'feature_selection_results.csv', index=False)
print("✓ Saved: feature_selection_results.csv")

# Export RFECV performance curve
rfecv_performance_df = pd.DataFrame({
    'N_Features': list(range(1, len(selector.cv_results_['mean_test_score']) + 1)),
    'CV_R2': selector.cv_results_['mean_test_score']
})
rfecv_performance_df.to_csv(output_dir / 'rfecv_performance_curve.csv', index=False)
print("✓ Saved: rfecv_performance_curve.csv")

# Export feature importance
importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
print("✓ Saved: feature_importance.csv")

# Export SHAP values summary
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0),
    'Mean_SHAP': shap_values.mean(axis=0),
    'Std_SHAP': shap_values.std(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)
shap_importance_df.to_csv(output_dir / 'shap_feature_importance.csv', index=False)
print("✓ Saved: shap_feature_importance.csv")

# Export per-row SHAP values for temporal analysis
shap_columns = [f"SHAP_{name}" for name in feature_names]
shap_values_df = pd.DataFrame(shap_values_full, columns=shap_columns)
shap_values_df.insert(0, 'Date', X_test_reset_full['Date'].values)
shap_values_df.insert(1, 'Year', X_test_reset_full['Date'].dt.year.values)
shap_values_df.insert(2, 'Month', X_test_reset_full['Date'].dt.month.values)
shap_values_df.to_csv(output_dir / 'shap_values.csv', index=False)
print("✓ Saved: shap_values.csv")

# Export predictions with metadata
predictions_df = pd.DataFrame({
    'Date': X_test_selected['Date'].values,
    'Observed_NEE': y_test.values,
    'Predicted_NEE': y_pred_test,
    'Residual': residuals_test.values,
    'Abs_Residual': np.abs(residuals_test.values)
})
for pred_name in selected_features:
    predictions_df[pred_name] = X_test_selected[pred_name].values
predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)
print("✓ Saved: test_predictions.csv")

# Export default RF parameters used
default_params_df = pd.DataFrame([
    {'Parameter': 'n_estimators', 'Value': 100},
    {'Parameter': 'max_depth', 'Value': 'None'},
    {'Parameter': 'min_samples_split', 'Value': 2},
    {'Parameter': 'min_samples_leaf', 'Value': 1},
    {'Parameter': 'max_features', 'Value': 'sqrt'},
    {'Parameter': 'Note', 'Value': 'Default RF parameters - hyperparameter tuning to be done later'}
])
default_params_df.to_csv(output_dir / 'rf_default_parameters.csv', index=False)
print("✓ Saved: rf_default_parameters.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings:")
print(f"  • Best CV R²: {cv_best_score:.4f}")
print(f"  • Test R²: {test_r2:.4f}")
print(f"  • Test RMSE: {test_rmse:.4f}")
print(f"\nTop 3 Most Important Features (SHAP):")
for i in range(min(3, len(shap_importance_df))):
    row = shap_importance_df.iloc[i]
    print(f"  {i+1}. {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")

print("\nOutput Files Generated:")
print("  • 8 visualization plots (PNG)")
print("  • 5 data files (CSV)")
print("\nAll results saved to current directory.")
print("="*70)

# ============================================================================
# 8. FINAL MODEL ON FULL DATASET: PREDICTIONS + SHAP EXPORT
# ============================================================================

print("\n[Final] Fitting final model on full dataset and exporting SHAP values...")

# Create full dataset with selected features only
X_full_selected = X[selected_features + ['Date']]

# Refit the best pipeline on all available cleaned data (with selected features)
best_pipeline.fit(X_full_selected, y)

# Predict NEE for the full dataset
full_pred = best_pipeline.predict(X_full_selected)

# SHAP values for all rows in the full dataset
X_full_transformed = best_pipeline.named_steps['pre'].transform(X_full_selected)
rf_model_full = best_pipeline.named_steps['rf']
explainer_full = shap.TreeExplainer(rf_model_full)
shap_values_full_dataset = explainer_full.shap_values(X_full_transformed)

# Prepare raw full-data features with seasonal terms for export
X_full_reset = X_full_selected.reset_index(drop=True)
X_full_raw = X_full_reset[selected_features].copy()
X_full_raw = X_full_raw[feature_names]

# Export full-dataset predictions + SHAP values
full_shap_columns = [f"SHAP_{name}" for name in feature_names]
full_predictions_shap_df = pd.DataFrame(shap_values_full_dataset, columns=full_shap_columns)
full_predictions_shap_df.insert(0, 'Date', X_full_reset['Date'].values)
full_predictions_shap_df.insert(1, 'Year', X_full_reset['Date'].dt.year.values)
full_predictions_shap_df.insert(2, 'Month', X_full_reset['Date'].dt.month.values)
full_predictions_shap_df.insert(3, 'Observed_NEE', y.reset_index(drop=True).values)
full_predictions_shap_df.insert(4, 'Predicted_NEE', full_pred)
full_predictions_shap_df.to_csv(output_dir / 'full_dataset_predictions_shap_values.csv', index=False)

print("✓ Saved: full_dataset_predictions_shap_values.csv")
