import pandas as pd
import joblib

def test_model_predicts_positive_price():
    model = joblib.load("house_price_model.pkl")

    house = pd.DataFrame({
        "area": [1100],
        "bedrooms": [2]
    })

    prediction = model.predict(house)[0]

    assert prediction > 0

def test_model_predicts_multiple_houses():
    model = joblib.load("house_price_model.pkl")

    houses = pd.DataFrame({
        "area": [800, 1200, 1800],
        "bedrooms": [2, 3, 4]
    })

    predictions = model.predict(houses)

    assert len(predictions) == 3