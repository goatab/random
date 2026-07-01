from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (NO SCALING NEEDED)
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

print("R2:", r2_score(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))

# Predict on real sample
new_sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(new_sample)

print("Prediction:", prediction)