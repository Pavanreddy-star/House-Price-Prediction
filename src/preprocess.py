import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Selecting relevant features
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    target = 'SalePrice' if 'SalePrice' in df.columns else None

    # Drop rows with missing values
    df = df[features + ([target] if target else [])].dropna()

    # Normalize numerical features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features, target

