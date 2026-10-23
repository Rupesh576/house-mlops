
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = {
    "area": [500, 700, 800, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [1, 1, 2, 2, 3, 3, 4, 4],
    "price": [20, 25, 35, 40, 55, 65, 80, 90]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Test MAE: ₹{mae:.2f} lakh")

# Fail the pipeline if the error is too high.
if mae > 15:
    raise ValueError("Model accuracy check failed!")

joblib.dump(model, "house_price_model.pkl")
print("Model trained, evaluated, and saved!")
