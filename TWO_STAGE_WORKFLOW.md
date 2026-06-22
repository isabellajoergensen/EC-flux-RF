# Two-Stage Modeling Workflow

## Overview

This workflow uses a two-stage approach to build optimized Random Forest models:
- **Stage 1**: Feature selection using RFECV
- **Stage 2**: Hyperparameter tuning with selected features

## Scripts

### Stage 1: Feature Selection (RFE)
- **ZaH**: `NEE_RFE_ZaH.py`
- **ZaF**: `NEE_RFE_ZaF.py`

**Purpose**: Use Recursive Feature Elimination with Cross-Validation (RFECV) to identify the most important features.

**Model**: Random Forest with **default parameters**
- n_estimators: 100
- max_depth: None (unlimited)
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: 'sqrt'

**Outputs**:
- `feature_selection_results.csv` - List of selected/removed features
- Model performance with default RF parameters
- SHAP analysis for selected features

### Stage 2: Hyperparameter Tuning (HPT)
- **ZaH**: `NEE_HPT_ZaH.py`
- **ZaF**: `NEE_HPT_ZaF.py`

**Purpose**: Optimize Random Forest hyperparameters with a customizable feature list.

**Feature Selection**: 
- Simple feature list defined at the top of the script
- Default list based on Stage 1 RFE results
- Directly edit the list to add or remove features
- No complex loading logic - just modify the Python list

**Hyperparameter Search Space**:
- n_estimators: 100-500
- max_depth: [10, 20, 30, 40, 50, None]
- min_samples_split: 2-20
- min_samples_leaf: 1-10
- max_features: ['sqrt', 'log2', 0.3, 0.5, 0.7]
- max_samples: 0.6-1.0

**Method**: RandomizedSearchCV with 100 iterations

**Outputs** (with `_HPT` suffix):
- `best_hyperparameters_HPT.csv` - Optimized hyperparameters
- `stage_comparison.csv` - Comparison between Stage 1 and Stage 2 models
- Updated model performance metrics, SHAP analysis, and visualizations

## Customizing Features in Stage 2

Stage 2 scripts have a simple `selected_features` list at the top that you can directly edit:

```python
selected_features = [
    # Seasonal (cyclical)
    'sin_doy',
    'cos_doy',
    
    # Snow and soil moisture
    'DSSM',
    
    # Vegetation
    'NDVI_nonSR_Median',
    'NDVI_nonSR_Max',
    
    # ... add or remove features here
]
```

**To customize**: Simply edit this list by adding or removing feature names. The default list is based on Stage 1 RFE recommendations.

## Data Leakage Prevention ✓

### Question: Is there data leakage between stages?

**Answer: NO - There is NO data leakage!**

### Why it's safe:

1. **Same Train-Test Split**: Both stages use:
   - Same `random_state=42`
   - Same `test_size=0.2` (80/20 split)
   - Same stratification strategy (by year)
   
   → This guarantees **identical train-test splits** in both scripts

2. **Test Data Never Used for Training**:
   - **Stage 1**: RFE performs cross-validation **only on training data**
   - **Stage 2**: Hyperparameter tuning performs cross-validation **only on training data**
   - The 20% test set remains completely untouched during both RFE and hyperparameter tuning

3. **What Each Stage Sees**:
   ```
   Stage 1 (RFE):
   ├── Training data (80%) → Used for feature selection with CV
   └── Test data (20%) → ONLY used for final evaluation
   
   Stage 2 (HPT):
   ├── Training data (80%) → Used for hyperparameter tuning with CV
   │                         (with features from Stage 1)
   └── Test data (20%) → ONLY used for final evaluation
   ```

4. **Cross-Validation Strategy**:
   - Both stages use **GroupKFold CV** on the training set
   - Groups are defined by year
   - Each CV fold holds out entire years from training
   - Test set is never part of any CV fold

### Key Principle

The test data is used **only for evaluation**, never for:
- Feature selection (Stage 1, if run)
- Hyperparameter tuning (Stage 2)
- Any model training or optimization decisions

Whether you use Stage 1 for feature guidance or define features directly in Stage 2, the test set remains untouched during all training and optimization - ensuring a valid evaluation!

## Execution Order

1. **Optional - Run Stage 1** (for feature selection guidance):
   ```powershell
   python NEE_RFE_ZaH.py
   python NEE_RFE_ZaF.py
   ```

2. **Run Stage 2** (edit the feature list in the script as needed):
   ```powershell
   python NEE_HPT_ZaH.py
   python NEE_HPT_ZaF.py
   ```

Stage 2 uses a simple feature list that you can edit directly in the script. The default list is based on Stage 1 RFE results, but Stage 1 is not required to run Stage 2.

## Benefits of This Approach

1. **Computational Efficiency**: 
   - RFE with hyperparameter tuning would be very expensive
   - Separating them makes the workflow much faster

2. **Clear Workflow**:
   - Stage 1 (optional): Answer "Which features?" using data-driven RFE
   - Stage 2: Optimize "What hyperparameters?" with your chosen features

3. **Flexibility**:
   - Stage 1 provides statistical guidance
   - You have full control to adjust features in Stage 2
   - Stage 2 can run independently if you already know which features to use

4. **Reproducibility**:
   - Fixed random seeds ensure consistent results
   - Clear, simple feature lists are easy to document and share

## Expected Improvements

Stage 2 (HPT) should show improvements over Stage 1 (if you ran it):
- ✓ Higher or equal test R²
- ✓ Lower or equal test RMSE/MAE
- ✓ Better generalization (smaller train-test gap)
- ✓ More stable predictions across years

If you ran Stage 1, check the `stage_comparison.csv` file to see the improvements from hyperparameter tuning!
