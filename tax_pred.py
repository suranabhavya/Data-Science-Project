import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data_path = 'data/cleaned_property_Asses_data.csv'
df = pd.read_csv(data_path, low_memory=False)

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nColumns with log values:")
log_columns = [col for col in df.columns if col.startswith('LOG_')]
print(log_columns)

# Check for missing values
missing_values = df.isnull().sum()
print("\nColumns with missing values:")
print(missing_values[missing_values > 0])

# Define target variable and feature selection
target = 'LOG_GROSS_TAX'
y = df[target].copy()

# Select features for prediction (excluding non-predictive columns)
exclude_cols = ['PID', 'GIS_ID', 'ST_NUM', 'ST_NAME', 'CITY', 'ZIP_CODE', 'OWNER', 
                'MAIL_STREET_ADDRESS', 'MAIL_CITY', 'MAIL_STATE', 'MAIL_ZIP_CODE',
                'GROSS_TAX', target]  # Exclude target and its non-log version

# Use logarithmic values for numerical features when available
features = [col for col in df.columns if col not in exclude_cols]
X = df[features].copy()

# Remove rows where target is missing
mask = ~y.isna()
X = X[mask]
y = y[mask]

print(f"\nAfter removing rows with missing target values: {X.shape}")

# Handle categorical variables (convert to numeric)
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {len(categorical_cols)}")
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with imputer and scaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Apply preprocessing
X_train_processed = num_pipeline.fit_transform(X_train)
X_test_processed = num_pipeline.transform(X_test)

print(f"\nProcessed training data shape: {X_train_processed.shape}")

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
print("\n--- Model Performance ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_processed, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_processed)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

# Identify best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"\nBest performing model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")

# Feature importance for Random Forest (if it's the best model or just for insight)
if 'Random Forest' in models:
    rf_model = results['Random Forest']['model']
    
    # Create a DataFrame for feature importance
    if hasattr(rf_model, 'feature_importances_'):
        feature_importances = pd.DataFrame(
            rf_model.feature_importances_,
            index=X_train.columns,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        # Show top 15 features
        print("\nTop 15 most important features:")
        print(feature_importances.head(15))

# Visualize actual vs predicted values for best model
plt.figure(figsize=(10, 6))
y_pred_best = results[best_model_name]['predictions']

# Convert from log values back to original scale for better interpretability
y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred_best)

plt.scatter(y_test_exp, y_pred_exp, alpha=0.5)
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'r--')
plt.xlabel('Actual Gross Tax')
plt.ylabel('Predicted Gross Tax')
plt.title(f'Actual vs Predicted Gross Tax Values using {best_model_name}')
plt.savefig('images/tax_prediction_results.png')

# Tune the best model with GridSearchCV if it's one of these types
if best_model_name == 'Ridge Regression':
    print("\nFine-tuning Ridge Regression...")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_processed, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best R²: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
    
elif best_model_name == 'Lasso Regression':
    print("\nFine-tuning Lasso Regression...")
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_processed, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best R²: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
    
elif best_model_name == 'Random Forest':
    print("\nFine-tuning Random Forest...")
    param_grid = {
        'n_estimators': [50],
        'max_depth': [10],
        'min_samples_split': [5]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=2, scoring='r2')
    grid_search.fit(X_train_processed, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best R²: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_

# Function to make predictions on new data
def predict_gross_tax(new_data):
    """
    Makes predictions on new data using the best trained model.
    
    Parameters:
    new_data (DataFrame): Data in the same format as training data
    
    Returns:
    numpy array: Predicted gross tax values (in original scale, not log)
    """
    # Ensure new_data has all categorical columns converted to dummies
    new_data = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)
    
    # Align columns with training data
    for col in X_train.columns:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[X_train.columns]
    
    # Apply preprocessing pipeline
    new_data_processed = num_pipeline.transform(new_data)
    
    # Predict log values
    log_predictions = best_model.predict(new_data_processed)
    
    # Convert back to original scale
    predictions = np.exp(log_predictions)
    
    return predictions

print("\nModel training and evaluation complete.")
