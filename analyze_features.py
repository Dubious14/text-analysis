import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Load the features data ===
df = pd.read_csv("clear_features.csv")

# === Basic Information ===
print("Loaded data:")
print(df.info())
print(df.describe())

# === Correlation Matrix ===
print("Correlation with BT_score:")
print(df.corr(numeric_only=True)['BT_score'].sort_values(ascending=False))

# === Plot each feature vs BT_score ===
feature_cols = [col for col in df.columns if col not in ['BT_score', 'SE', 'id']]

for feature in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[feature], y=df['BT_score'])
    plt.title(f"BT_score vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("BT_score")
    plt.grid(True)

    # Optional: simple linear regression fit
    model = LinearRegression()
    X = df[[feature]]
    y = df['BT_score']
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Plot regression line
    plt.plot(X, y_pred, color='red', label=f"R² = {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Pairplot (scatter matrix) ===
# Combine features + target into one plot
pairplot_cols = feature_cols + ['BT_score']

print("Generating pairplot... (can take a few seconds)")

sns.pairplot(df[pairplot_cols], corner=True, diag_kind='kde')
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()
