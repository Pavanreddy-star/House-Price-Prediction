import pandas as pd
import joblib
from preprocess import preprocess_data

# Load trained model (use absolute path)
model_path = r"C:\Users\PranithaReddy\Desktop\House-Price-Prediction\models\house_price_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first using `python src/train.py`.")

# Load test data
df_test = pd.read_csv(r"C:\Users\PranithaReddy\Desktop\House-Price-Prediction\data\test.csv")  # Use absolute path
df_test, features, _ = preprocess_data(df_test)

# Predict house prices
predictions = model.predict(df_test[features])

# Save predictions
output_path = r"C:\Users\PranithaReddy\Desktop\House-Price-Prediction\data\submission.csv"
output = pd.DataFrame({'Id': df_test.index, 'SalePrice': predictions})
output.to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")

