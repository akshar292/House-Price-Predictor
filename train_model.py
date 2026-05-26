import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("house_price_dataset.csv")

# Input
X = df[["Area_sqft", "Bedrooms", "Age_years"]]

# Output
Y = df["Price"]

# Model
model = LinearRegression()

# Train
model.fit(X, Y)

# Save model
pickle.dump(model, open("house_price_model.pkl", "wb"))

print("Model Trained Successfully")