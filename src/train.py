import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import preprocess_data
from model import build_model

# Load dataset
df = pd.read_csv('data/train.csv')

# Preprocess data
df, features, target = preprocess_data(df)  # Ensure features & target are correctly returned

# Split dataset into training and testing
X = df[features]   # Features
y = df[target]     # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
model = build_model()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# Save the trained model
import joblib
joblib.dump(model, 'models/house_price_model.pkl')

