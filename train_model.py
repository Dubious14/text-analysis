import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

# --- Load features from CSV ---
data = pd.read_csv("clear_features.csv")

# Fix bad SEs
data = data.copy()
data['SE'] = data['SE'].replace(0, np.nan)
data = data.dropna(subset=['SE'])

X = data.drop(columns=['BT_score', 'SE', 'id'])
y = data['BT_score']
weights = 1 / (data['SE'] ** 2)
# --- Find best lambda (alpha) using CV ---
alphas = np.logspace(-4, 3, 50)  # Search from 10^-4 to 10^3

model_cv = RidgeCV(alphas=alphas, scoring='neg_root_mean_squared_error', store_cv_values=True)
model_cv.fit(X, y, sample_weight=weights)

best_alpha = model_cv.alpha_
print(f"✅ Best alpha (lambda) found by CV: {best_alpha:.6f}")

# --- Retrain Ridge on ALL data using best alpha ---
model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
model.fit(X, y, ridge__sample_weight=weights)

# --- Predict back on train set (for evaluation) ---
y_pred = model.predict(X)

# --- Metrics ---
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

print("\n📈 Model Evaluation on Full Training Set:")
print(f"R²       : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")

# --- Additional analysis with Statsmodels (for F-statistic etc.) ---
# We need to re-fit manually with standardized features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_with_const = sm.add_constant(X_scaled)
ols_model = sm.OLS(y, X_with_const).fit()

print("\n📊 Full OLS-like Summary:")
print(ols_model.summary())

# --- Coefficients ---
ridge_final = model.named_steps['ridge']
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": ridge_final.coef_
}).sort_values(by="Coefficient", key=np.abs, ascending=False)

print("\n📋 Coefficients (sorted by absolute value):")
print(coef_df)
