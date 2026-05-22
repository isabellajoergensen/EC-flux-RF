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
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

# Configure matplotlib for better plots
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# --- Seasonal transformer --- #
class SeasonalFeatures(BaseEstimator, TransformerMixin):
    """Transform datetime to cyclical seasonal features"""
    def __init__(self, datetime_col='Date'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2 = X.copy()
        doy = X2[self.datetime_col].dt.dayofyear
        X2['sin_doy'] = np.sin(2 * np.pi * doy / 365)
        X2['cos_doy'] = np.cos(2 * np.pi * doy / 365)
        return X2.drop(columns=[self.datetime_col])

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

print("="*70)
print("NEE FLUX ANALYSIS WITH SHAP INTERPRETATION")
print("="*70)
print("\n[1/6] Loading and preparing data...")

# Output configuration
site_ID = "ZaH"
period = "_GS" # "" for full period, "_GS" for growing season and "_fieldseason" for field season only 
output_dir = Path(__file__).resolve().parent / "output" / site_ID
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
target_col = 'NEE_U50_f_mean'
predictors = ['Snow_Cover_Percentage',
              'NDVI_Median',
              'DSSM',
              'rH_f_mean',
              'Tair_f_mean',
              'Tair_f_min',
              'Tair_f_max',
              'Rg_f_mean',
              'Rg_f_min',
              'Rg_f_max',
              'VPD_f_mean',
              #'SWC_5cm', 
              'SWC_10cm',
              #'TS_5cm',
              'TS_10cm',
              'precipitation_rate']



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

print(f"\nStrategy: Train/Test Split + Group K-Fold CV")
print("  • 20% held out for final testing")
print("  • Group K-Fold CV on training data\n")

# Hold out 20% for final testing
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

n_cv_splits = min(5, groups_train.nunique())
if n_cv_splits < 2:
    raise ValueError("Need at least 2 unique years for GroupKFold.")
cv_strategy = GroupKFold(n_splits=n_cv_splits)
n_tune_iter = 50

# ============================================================================
# 3. HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# ============================================================================

print("\n[3/6] Performing hyperparameter tuning...")

# Build preprocessing pipeline
preproc = ColumnTransformer([
    ('season', SeasonalFeatures(datetime_col='Date'), ['Date']),
    ('scale', StandardScaler(), predictors)
], remainder='drop')

# Create base pipeline
base_pipeline = Pipeline([
    ('pre', preproc),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Define hyperparameter search space
# Ranges are constrained toward regularisation to reduce overfitting:
#   - max_depth: no unlimited trees; cap at 20 to prevent memorisation
#   - min_samples_leaf: minimum 5 so leaves represent meaningful patterns
#   - min_samples_split: minimum 10 to avoid splitting on noise
param_distributions = {
    'rf__n_estimators': randint(100, 400),
    'rf__max_depth': [5, 8, 10, 15, 20],
    'rf__min_samples_split': randint(10, 40),
    'rf__min_samples_leaf': randint(5, 25),
    'rf__max_features': ['sqrt', 'log2', 0.3, 0.5],
    'rf__bootstrap': [True]
}

# Setup cross-validation for hyperparameter tuning
tune_cv = cv_strategy.split(X_train, y_train, groups_train)
X_tune, y_tune, groups_tune = X_train, y_train, groups_train

# Randomized search
random_search = RandomizedSearchCV(
    base_pipeline,
    param_distributions,
    n_iter=n_tune_iter,
    cv=tune_cv,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print(f"Starting randomized search with {n_tune_iter} iterations and {n_cv_splits} CV folds...")
random_search.fit(X_tune, y_tune)

print("\n--- Best Parameters ---")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")
print(f"\nBest CV R²: {random_search.best_score_:.4f}")

# Get best model
best_pipeline = random_search.best_estimator_

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================

print("\n[4/6] Evaluating model performance...")

cv_scores = cross_val_score(
    best_pipeline,
    X_train,
    y_train,
    cv=cv_strategy.split(X_train, y_train, groups_train),
    scoring='r2',
    n_jobs=-1
)

print(f"\nCross-Validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"R² per fold: {np.round(cv_scores, 4)}")

# Test set evaluation
y_pred_train = best_pipeline.predict(X_train)
y_pred_test = best_pipeline.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n--- Model Performance ---")
print(f"{'Metric':<15} {'Train':<15} {'Test':<15}")
print("-" * 45)
print(f"{'R²':<15} {train_r2:<15.4f} {test_r2:<15.4f}")
print(f"{'RMSE':<15} {train_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"{'MAE':<15} {train_mae:<15.4f} {test_mae:<15.4f}")

year_metrics = None

# ============================================================================
# 5. SHAP ANALYSIS - PREDICTOR IMPORTANCE AND RELATIONSHIPS
# ============================================================================

print("\n[5/6] Performing SHAP analysis...")

# Prepare feature names (after transformation)
feature_names = ['sin_doy', 'cos_doy'] + predictors

# Transform data for SHAP
X_train_transformed = best_pipeline.named_steps['pre'].transform(X_train)
X_test_transformed = best_pipeline.named_steps['pre'].transform(X_test)

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

# Prepare raw (unscaled) data for SHAP plots
X_test_reset = X_test.iloc[sample_idx].reset_index(drop=True)
doy = X_test_reset['Date'].dt.dayofyear
X_test_raw = X_test_reset[predictors].copy()
X_test_raw['sin_doy'] = np.sin(2 * np.pi * doy / 365)
X_test_raw['cos_doy'] = np.cos(2 * np.pi * doy / 365)
X_test_raw = X_test_raw[feature_names]

# Prepare raw (unscaled) data for full test set (for export)
X_test_reset_full = X_test.reset_index(drop=True)
doy_full = X_test_reset_full['Date'].dt.dayofyear
X_test_raw_full = X_test_reset_full[predictors].copy()
X_test_raw_full['sin_doy'] = np.sin(2 * np.pi * doy_full / 365)
X_test_raw_full['cos_doy'] = np.cos(2 * np.pi * doy_full / 365)
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

# --- Plot 3: Feature Importance (from Random Forest) ---
rf_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importances
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Random Forest Feature Importance')
ax.grid(True, alpha=0.3, axis='x')

# Color bars
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
plt.savefig(output_dir / '03_feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_feature_importance_rf.png")
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
    'CV_Mean': [cv_scores.mean(), np.nan, np.nan],
    'CV_Std': [cv_scores.std(), np.nan, np.nan]
})
metrics_df.to_csv(output_dir / 'model_performance_metrics.csv', index=False)
print("✓ Saved: model_performance_metrics.csv")

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
    'Date': X_test['Date'].values,
    'Observed_NEE': y_test.values,
    'Predicted_NEE': y_pred_test,
    'Residual': residuals_test.values,
    'Abs_Residual': np.abs(residuals_test.values)
})
for i, pred_name in enumerate(predictors):
    predictions_df[pred_name] = X_test[pred_name].values
predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)
print("✓ Saved: test_predictions.csv")

# Export best hyperparameters
best_params_df = pd.DataFrame([
    {'Parameter': k, 'Value': v} for k, v in random_search.best_params_.items()
])
best_params_df.to_csv(output_dir / 'best_hyperparameters.csv', index=False)
print("✓ Saved: best_hyperparameters.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings:")
print(f"  • Best CV R²: {random_search.best_score_:.4f}")
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

# Refit the best pipeline on all available cleaned data
best_pipeline.fit(X, y)

# Predict NEE for the full dataset
full_pred = best_pipeline.predict(X)

# SHAP values for all rows in the full dataset
X_full_transformed = best_pipeline.named_steps['pre'].transform(X)
rf_model_full = best_pipeline.named_steps['rf']
explainer_full = shap.TreeExplainer(rf_model_full)
shap_values_full_dataset = explainer_full.shap_values(X_full_transformed)

# Prepare raw full-data features with seasonal terms for export
X_full_reset = X.reset_index(drop=True)
doy_all = X_full_reset['Date'].dt.dayofyear
X_full_raw = X_full_reset[predictors].copy()
X_full_raw['sin_doy'] = np.sin(2 * np.pi * doy_all / 365)
X_full_raw['cos_doy'] = np.cos(2 * np.pi * doy_all / 365)
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
