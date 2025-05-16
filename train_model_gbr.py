# train_model_gbr.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load features ---
data = pd.read_csv("clear_features.csv")

# --- Clean data ---
data = data.copy()
data['SE'] = data['SE'].replace(0, np.nan)
data = data.dropna(subset=['SE'])

X = data.drop(columns=['BT_score', 'SE', 'id'])
y = data['BT_score']

# --- Train-test split (optional, 90% train / 10% test) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# --- Train Gradient Boosting Regressor ---
gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gbr.fit(X_train, y_train)

# --- Evaluate on test set ---
y_pred = gbr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation on Test Set:")
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")

# --- Feature Importance ---
importances = gbr.feature_importances_
feat_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n📋 Feature Importances:")
print(feat_importance)

# --- Plot Feature Importances ---
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_importance)
plt.title("Feature Importances (GBR)")
plt.grid(True)
plt.tight_layout()
plt.show()
