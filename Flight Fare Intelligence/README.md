# Flight Fare Intelligence

A Streamlit-based machine learning project for predicting flight fares from route, airline, schedule, duration, stop count, and booking-window features.

The project provides a complete workflow:

- Load a flight fare dataset from CSV or Excel.
- Automatically detect and standardize common flight dataset columns.
- Explore dataset quality, fare distribution, airline/category trends, and route behavior.
- Train multiple regression models and save the best pipeline.
- Predict a fare from user-entered flight details.
- Show base model fare, time-adjusted fare, and simulated live fare.

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Main Streamlit application. Run this file to use the web interface. |
| `train.py` | Trains Linear Regression, Random Forest, and XGBoost models, then saves the best model. |
| `preprocess.py` | Column detection, data cleaning, feature engineering, and sklearn pipeline construction. |
| `utils.py` | Dataset loading, model persistence, metrics, price cleaning, demo data, and helper functions. |
| `Flight_Fare.xlsx` | Included flight fare dataset for training or demonstration. |
| `Flight_Fare_Prediction_Using_Machine_Learning.ipynb` | Notebook version of the project workflow. |
| `requirements.txt` | Python package dependencies. |
| `PROJECT_STRUCTURE.md` | Notes about the project structure. |
| `FACULTY_DEMO_GUIDE.md` | Guide for presenting or demonstrating the project. |

Generated model files are saved in a `models/` folder:

```text
models/
  flight_model.pkl
  model_info.pkl
```

## Features

- Dataset upload support for `.csv`, `.xlsx`, and `.xls` files.
- Automatic role detection for columns such as airline, source, destination, journey date, departure time, arrival time, duration, stops, route, and fare.
- Price column cleaning for numeric and formatted fare values.
- Feature engineering for date, time, duration, stops, route, source-destination pairs, distance, weekend flags, booking window, and yearly trend.
- Model comparison using R2, MAE, and RMSE.
- Saved inference pipeline using `joblib`.
- Interactive Streamlit pages for data loading, insights, training, and prediction.

## Requirements

- Python 3.8 or newer
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- matplotlib
- seaborn
- joblib
- openpyxl

## How to Run

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Recommended Workflow

1. Open the app with `streamlit run app.py`.
2. Go to the `Data Loading` page.
3. Upload `Flight_Fare.xlsx` or another compatible CSV/Excel dataset.
4. Review the detected dataset profile and preview.
5. Click `Train model on current dataset`.
6. Check model metrics after training completes.
7. Go to `Insights` to review fare and route patterns.
8. Go to `Prediction` and enter flight details to estimate the fare.

The app also includes a small built-in demo dataset. Use `Load default demo dataset` if you only want to test the interface quickly.

## Training from the Command Line

Train on the included Excel dataset:

```bash
python train.py --data Flight_Fare.xlsx
```

Train on another dataset:

```bash
python train.py --data path/to/your_dataset.csv
```

Train on the built-in demo rows:

```bash
python train.py
```

After training, the best model pipeline and metadata are saved in `models/`.

## Supported Dataset Columns

The project can detect common variations of these fields:

| Required field | Examples |
| --- | --- |
| Airline | `Airline`, `Carrier`, `Airline_Name` |
| Source | `Source`, `From`, `Origin`, `Departure_City` |
| Destination | `Destination`, `To`, `Dest`, `Arrival_City` |
| Journey date | `Date_of_Journey`, `Date`, `Journey_Date`, `Travel_Date` |
| Departure time | `Dep_Time`, `Departure_Time`, `Departure`, `DepTime` |
| Duration | `Duration`, `Flight_Duration`, `Travel_Time`, `time_taken` |
| Price/Fare | `Price`, `Fare`, or similar fare/price column names |

Optional fields such as `Arrival_Time`, `Total_Stops`, `Route`, and `Additional_Info` improve feature quality when available.

## Models Used

`train.py` compares these regression models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The model with the highest R2 score on the test split is selected and saved. Evaluation metrics include:

- R2 score
- Mean Absolute Error
- Root Mean Squared Error

## Application Pages

### Data Loading

Upload or load a dataset, inspect column detection, preview data, view fare statistics, and train/retrain the model.

### Insights

Review dataset overview, column quality, fare insights, category counts, and route-level patterns.

### Prediction

Enter flight details and generate:

- Base ML prediction
- Time-adjusted fare
- Simulated live fare with demand adjustment

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt`. |
| No model found | Train a model from the Data Loading page or run `python train.py --data Flight_Fare.xlsx`. |
| Excel file cannot be read | Ensure `openpyxl` is installed from `requirements.txt`. |
| Dataset is rejected | Check that required columns for airline, source, destination, date, departure time, duration, and price are present. |
| Streamlit port is busy | Run `streamlit run app.py --server.port 8502`. |

## Project Status

This project is ready for local demonstration, academic submission, and viva/faculty presentation. The core code is modular, with separate files for the app, training, preprocessing, and utilities.
