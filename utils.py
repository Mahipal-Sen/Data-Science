"""
=============================================================================
UTILITY FUNCTIONS MODULE
=============================================================================
Role: Core utility functions for the entire system
Purpose:
  - Model I/O: Save and load trained models and metadata
  - Dataset Loading: Handle CSV/Excel file imports
  - Metrics Calculation: Compute R², MAE, RMSE for model evaluation
  - Data Cleaning: Clean price columns with various formats
  - Price Adjustments: Apply time-based and demand-based price adjustments
  - Logging Setup: Configure logging for debugging and monitoring
  - Demo Data: Provide sample dataset for demonstrations

Key Functions:
  - save_pipeline() / load_pipeline(): Model persistence
  - save_metadata() / load_metadata(): Metadata storage for inference
  - calculate_regression_metrics(): Evaluate model performance
  - clean_price_column(): Handle price data in various formats
  - adjust_price_for_time(): Apply time-decay pricing model
  - get_live_price(): Simulate real-time pricing with demand factors
  - get_demo_dataset(): Load sample flights.csv for demos

Usage:
  - Imported by all other modules (train.py, app.py, preprocess.py)
=============================================================================
"""

import logging
import os
import re
import warnings
from datetime import datetime
from io import BytesIO
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'flight_model.pkl')
DEFAULT_INFO_PATH = os.path.join(MODELS_DIR, 'model_info.pkl')
PRICE_ALIASES = [
    'price', 'fare', 'ticket_price', 'ticketprice', 'ticket price',
    'fare_price', 'fare price', 'amount', 'cost', 'total_price', 'total price',
    'total_fare', 'ticket_amount', 'ticket cost'
]


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logging():
    ensure_directory(LOGS_DIR)
    log_path = os.path.join(LOGS_DIR, 'project.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Logging initialized.')


def save_pipeline(pipeline, path: str = DEFAULT_MODEL_PATH):
    ensure_directory(MODELS_DIR)
    joblib.dump(pipeline, path)
    logging.info(f'Saved model pipeline to {path}')


def load_pipeline(path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    pipeline = joblib.load(path)
    return pipeline


def save_metadata(metadata: dict, path: str = DEFAULT_INFO_PATH):
    ensure_directory(MODELS_DIR)
    joblib.dump(metadata, path)
    logging.info(f'Saved model metadata to {path}')


def load_metadata(path: str = DEFAULT_INFO_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Metadata file not found: {path}')
    return joblib.load(path)


def calculate_regression_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))
    total_variance = np.sum(np.square(y_true - np.mean(y_true)))
    residual_variance = np.sum(np.square(errors))
    r2 = 0.0 if total_variance == 0 else 1 - (residual_variance / total_variance)
    return {'r2': float(r2), 'mae': float(mae), 'rmse': float(rmse)}


def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    elif path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    else:
        raise ValueError('Unsupported file format. Upload a CSV or Excel file.')
    return df


def validate_dataset_for_prediction(df: pd.DataFrame, roles: dict) -> Tuple[bool, str]:
    """Validate dataset for prediction."""
    if df.empty:
        return False, "Dataset is empty"

    required_roles = ['airline', 'source', 'destination', 'date', 'dep_time', 'duration']
    missing_roles = [role for role in required_roles if roles.get(role) is None]

    if missing_roles:
        return False, f"Missing required column mappings: {', '.join(missing_roles)}"

    return True, "Dataset is valid for prediction"


def make_predictions(pipeline: Any, input_data: pd.DataFrame) -> np.ndarray:
    """Make predictions using the trained pipeline."""
    try:
        predictions = pipeline.predict(input_data)
        return predictions
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise ValueError(f"Failed to make predictions: {str(e)}")


def _find_price_column(df: pd.DataFrame) -> Optional[str]:
    lower_columns = {str(col).strip().lower(): col for col in df.columns}
    for alias in PRICE_ALIASES:
        if alias in lower_columns:
            return lower_columns[alias]
    for col in df.columns:
        normalized = re.sub(r'[^a-z0-9]', '', str(col).lower())
        if normalized in PRICE_ALIASES:
            return col
    return None


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset price column so it is always numeric."""
    if df is None or df.empty:
        raise ValueError('Input dataframe is empty.')

    price_col = _find_price_column(df)
    if price_col is None:
        raise ValueError('Price column not found. Expected one of: price, Price, Fare, ticket_price, amount, cost.')

    cleaned_df = df.copy()
    raw_series = cleaned_df[price_col].astype(str)
    cleaned_values = raw_series.str.replace(r'[\,\s₹$£€]', '', regex=True)
    cleaned_values = cleaned_values.str.replace(r'[^\d\.-]', '', regex=True)
    numeric_price = pd.to_numeric(cleaned_values, errors='coerce')

    cleaned_df[price_col] = numeric_price

    print(f"Price column detected: '{price_col}'")
    print('Datatype after cleaning:', cleaned_df[price_col].dtype)
    print('Null values after cleaning:', int(cleaned_df[price_col].isna().sum()))
    print('First 5 cleaned price values:', cleaned_df[price_col].head(5).tolist())

    if cleaned_df[price_col].isna().all():
        warnings.warn('All values in the price column are NaN after cleaning. Verify the input data formatting.')
        return cleaned_df

    cleaned_df = cleaned_df.loc[cleaned_df[price_col].notna()].copy()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df


def get_price_statistics(df: pd.DataFrame) -> dict:
    """Calculate mean, median, and mode for the cleaned price column."""
    if df is None or df.empty:
        raise ValueError('Input dataframe is empty. Cannot calculate price statistics.')

    price_col = _find_price_column(df)
    if price_col is None:
        raise ValueError('Price column not found. Expected one of: price, Price, Fare, ticket_price, amount, cost.')

    cleaned_df = clean_price_column(df)
    if cleaned_df.empty:
        warnings.warn('No valid numeric prices remain after cleaning. Statistics cannot be calculated.')
        return {'mean': None, 'median': None, 'mode': None}

    series = cleaned_df[price_col].dropna().astype(float)
    if series.empty:
        warnings.warn('Price column has no numeric values after cleaning. Statistics cannot be calculated.')
        return {'mean': None, 'median': None, 'mode': None}

    print('Datatype before statistics:', series.dtype)
    print('Null values before statistics:', int(series.isna().sum()))
    print('First 5 values before statistics:', series.head(5).tolist())

    mean_value = float(series.mean())
    median_value = float(series.median())
    mode_series = series.mode()
    mode_value = float(mode_series.iloc[0]) if not mode_series.empty else None

    return {
        'mean': mean_value,
        'median': median_value,
        'mode': mode_value
    }


def format_predictions(predictions: np.ndarray, currency: str = "₹") -> list:
    """Format predictions for display."""
    return [f"{currency}{pred:,.0f}" for pred in predictions]


def adjust_price_for_time(price: float, days_left: int, trend_factor: float) -> float:
    """Adjust a predicted flight price for booking window and yearly trend."""
    if price is None:
        raise ValueError('Base price cannot be None for time adjustment.')

    days_left = 0 if days_left is None else int(days_left)
    days_left = max(days_left, 0)
    trend_factor = 1.0 if trend_factor is None else float(trend_factor)

    adjusted_price = float(price)
    if days_left < 3:
        adjusted_price *= 1.3
    elif days_left > 20:
        adjusted_price *= 0.85

    adjusted_price *= trend_factor
    return float(adjusted_price)


def get_live_price(base_price: float, days_left: int, trend_factor: float, demand_factor: Optional[float] = None) -> float:
    """Return a simulated live price for a flight based on demand and time adjustments."""
    if demand_factor is None:
        demand_factor = float(np.random.uniform(0.9, 1.3))

    price = adjust_price_for_time(base_price, days_left, trend_factor)
    return float(price * demand_factor)


def get_demo_dataset() -> pd.DataFrame:
    return pd.DataFrame([
        {
            'Airline': 'IndiGo',
            'Source': 'Banglore',
            'Destination': 'New Delhi',
            'Date_of_Journey': '24/03/2019',
            'Dep_Time': '22:20',
            'Arrival_Time': '01:10',
            'Duration': '2h 50m',
            'Total_Stops': 'non-stop',
            'Price': 3897
        },
        {
            'Airline': 'Air India',
            'Source': 'Kolkata',
            'Destination': 'Banglore',
            'Date_of_Journey': '01/05/2019',
            'Dep_Time': '05:50',
            'Arrival_Time': '13:15',
            'Duration': '7h 25m',
            'Total_Stops': '2 stops',
            'Price': 7662
        },
        {
            'Airline': 'SpiceJet',
            'Source': 'Delhi',
            'Destination': 'Cochin',
            'Date_of_Journey': '12/04/2019',
            'Dep_Time': '09:00',
            'Arrival_Time': '12:30',
            'Duration': '3h 30m',
            'Total_Stops': '1 stop',
            'Price': 6218
        },
        {
            'Airline': 'Jet Airways',
            'Source': 'Mumbai',
            'Destination': 'Paris',
            'Date_of_Journey': '20/06/2019',
            'Dep_Time': '22:30',
            'Arrival_Time': '06:20',
            'Duration': '9h 50m',
            'Total_Stops': '2 stops',
            'Price': 5412
        }
    ])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> tuple[bytes, str]:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions_{timestamp}.csv'
    return buffer.getvalue(), filename


if __name__ == '__main__':
    setup_logging()
    print('Demo dataset loaded:')
    print(get_demo_dataset().head())
