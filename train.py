import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

data = {
    "area": [500, 700, 800, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [1, 1, 2, 2, 3, 3, 4, 4],
    "price": [20, 25, 35, 40, 55, 65, 80, 90]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "house_price_model.pkl")

print("Model trained and saved!")