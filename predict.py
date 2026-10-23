import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")

new_house = pd.DataFrame({
    "area": [1100],
    "bedrooms": [2]
})

prediction = model.predict(new_house)[0]

print(f"Predicted house price: ₹{prediction:.2f} lakh")