from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = pickle.load(open("house_price_model.pkl", "rb"))

# Input Structure
class HouseData(BaseModel):
    area_sqft: float
    bedrooms: int
    age_years: int

# Home Route
@app.get("/")
def home():
    return {"message": "House Price Predictor API Running"}

# Prediction Route
@app.post("/predict")
def predict(data: HouseData):

    prediction = model.predict([[
        data.area_sqft,
        data.bedrooms,
        data.age_years
    ]])

    return {
        "predicted_price": float(prediction[0])
    }