"""
Feature Correlation Matrix Analysis
Compute and visualize correlations between all predictors and target variable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
site_ID = "ZaF"
period = "_fieldseason"  # "" for full period, "_GS" for growing season, "_fieldseason" for field season
output_dir = Path(__file__).resolve().parent / "output" / site_ID
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("FEATURE CORRELATION MATRIX ANALYSIS")
print("="*70)

# Load data
flux_path = Path(__file__).resolve().parent / "data" / f"{site_ID}_complete{period}.csv"
data = pd.read_csv(flux_path, sep=',', parse_dates=['Date'])
data = data.sort_values('Date').reset_index(drop=True)

# Define target and predictors
target_col = 'NEE_U50_f'
predictors = [
    # Seasonal (cyclical)
    'sin_doy',
    'cos_doy',
    
    # Environmental conditions
    'Snow_Cover_Percentage',
    'DSSM',
    'D_SNOW',
    
    # Vegetation
    'NDVI_nonSR_Max',
    
    # Meteorology - current day
    'RH',
    'TA',
    'TA_min',
    'TA_max',
    'RG',
    'RG_min',
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
    'TS_lag1d',
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

# Compute cyclical seasonal features
data['sin_doy'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365)
data['cos_doy'] = np.cos(2 * np.pi * data['Date'].dt.dayofyear / 365)

# Drop rows with missing data
data_clean = data.dropna(subset=[target_col] + predictors).copy()

print(f"\nDataset: {site_ID}{period}")
print(f"Total observations: {len(data_clean)}")
print(f"Features: {len(predictors)}")

# Compute correlation matrix
all_features = [target_col] + predictors
corr_matrix = data_clean[all_features].corr()

# Save correlation matrix
corr_matrix.to_csv(output_dir / 'correlation_matrix.csv')
print(f"\n✓ Saved: correlation_matrix.csv")

# Extract correlations with target variable
target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
target_corr_df = pd.DataFrame({
    'Feature': target_corr.index,
    'Correlation': target_corr.values
})
target_corr_df.to_csv(output_dir / 'target_correlations.csv', index=False)
print(f"✓ Saved: target_correlations.csv")

# Print top correlations with target
print(f"\nTop 10 Features Correlated with {target_col}:")
print("-" * 50)
for i, (feat, corr) in enumerate(target_corr.head(10).items(), 1):
    print(f"{i:2d}. {feat:25s} r = {corr:+.3f}")

print(f"\nBottom 10 Features Correlated with {target_col}:")
print("-" * 50)
for i, (feat, corr) in enumerate(target_corr.tail(10).items(), 1):
    print(f"{i:2d}. {feat:25s} r = {corr:+.3f}")

# Create correlation heatmap
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Set ticks and labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.index)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(corr_matrix.index, rotation=0, fontsize=8)

# Add gridlines
ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(corr_matrix.index)) - 0.5, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

plt.title(f'Feature Correlation Matrix - {site_ID}{period}', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'correlation_matrix_full.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: correlation_matrix_full.png")

# Create target correlation bar plot
fig, ax = plt.subplots(figsize=(10, 12))
colors = ['green' if x > 0 else 'red' for x in target_corr.values]
ax.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
ax.set_yticks(range(len(target_corr)))
ax.set_yticklabels(target_corr.index, fontsize=9)
ax.set_xlabel('Correlation Coefficient', fontsize=11)
ax.set_title(f'Feature Correlations with {target_col} - {site_ID}{period}', fontsize=12, pad=15)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'target_correlations_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: target_correlations_barplot.png")

# Find highly correlated feature pairs (multicollinearity check)
print(f"\nHighly Correlated Feature Pairs (|r| > 0.8):")
print("-" * 70)
high_corr_pairs = []
for i in range(len(predictors)):
    for j in range(i+1, len(predictors)):
        corr_val = corr_matrix.iloc[i+1, j+1]  # +1 to skip target column
        if abs(corr_val) > 0.8:
            high_corr_pairs.append((predictors[i], predictors[j], corr_val))

if high_corr_pairs:
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"  {feat1:25s} <-> {feat2:25s}  r = {corr_val:+.3f}")
else:
    print("  No feature pairs with |r| > 0.8")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
