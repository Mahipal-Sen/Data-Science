"""
=============================================================================
MODEL TRAINING MODULE
=============================================================================
Role: Trains and evaluates multiple ML models for flight fare prediction
Purpose: 
  - Preprocesses flight data (standardization, feature engineering)
  - Trains three models: Linear Regression, Random Forest, XGBoost
  - Evaluates models using R², MAE, and RMSE metrics
  - Selects the best performing model
  - Saves trained pipeline and metadata for inference

Models Trained:
  1. Linear Regression: Fast baseline model
  2. Random Forest: Captures complex non-linear relationships
  3. XGBoost: Best overall performance (typically R² > 0.85)

Usage:
  - Called automatically by app.py when user uploads dataset
  - Can also be run standalone: python train.py --data flights.csv
=============================================================================
"""

import argparse
import logging
import os

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import preprocess
import utils


def train_model(df, save_model=True):
    df = utils.clean_price_column(df)
    validation, message = preprocess.validate_ready_for_training(df)
    if not validation:
        raise ValueError(message)

    standardized = preprocess.standardize_columns(df)
    roles = preprocess.infer_roles(standardized)
    X, y, metadata = preprocess.build_feature_dataframe(standardized, roles)
    if y is None or len(y) == 0:
        raise ValueError('Target variable could not be built after preprocessing.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def wrap_target_model(model):
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        )

    models = {
        'Linear Regression': wrap_target_model(LinearRegression()),
        'Random Forest': wrap_target_model(RandomForestRegressor(
            n_estimators=400,
            max_depth=20,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )),
        'XGBoost': wrap_target_model(XGBRegressor(
            random_state=42,
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            tree_method='hist',
            eval_metric='rmse',
            reg_alpha=1.5,
            reg_lambda=2.5,
            verbosity=0,
            n_jobs=-1
        ))
    }

    results = {}
    logging.info('Starting model training.')

    for name, model in models.items():
        pipeline = preprocess.build_training_pipeline(metadata, model)
        logging.info(f'Training {name}.')
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = utils.calculate_regression_metrics(y_test, y_pred)
        results[name] = {
            'pipeline': pipeline,
            'metrics': metrics
        }
        logging.info(f'{name} R2={metrics["r2"]:.4f}, MAE={metrics["mae"]:.2f}, RMSE={metrics["rmse"]:.2f}')

    best_name = max(results, key=lambda n: results[n]['metrics']['r2'])
    best_pipeline = results[best_name]['pipeline']
    best_metrics = results[best_name]['metrics']
    logging.info(f'Selected best model: {best_name}')

    if save_model:
        utils.save_pipeline(best_pipeline)
        # Save metadata properly with all required keys
        metadata_to_save = {
            'best_model': best_name,
            'metrics': best_metrics,
            'roles': roles,
            'numeric_features': metadata['numeric_features'],
            'categorical_low_cardinality': metadata['categorical_low_cardinality'],
            'categorical_high_cardinality': metadata['categorical_high_cardinality'],
            'feature_columns': metadata['feature_columns'],
            'target': metadata['target']
        }
        utils.save_metadata(metadata_to_save)

    full_pipeline = preprocess.build_training_pipeline(metadata, models[best_name])
    full_pipeline.fit(X, y)
    if save_model:
        utils.save_pipeline(full_pipeline)
        utils.save_metadata(metadata_to_save)
    
    return {
        'best_model': best_name,
        'metrics': best_metrics,
        'candidates': {name: info['metrics'] for name, info in results.items()},
        'pipeline': full_pipeline,
        'metadata': metadata
    }


def main():
    parser = argparse.ArgumentParser(description='Train flight fare prediction model on a CSV or Excel dataset.')
    parser.add_argument('--data', type=str, default=None, help='Path to training dataset file')
    args = parser.parse_args()

    utils.setup_logging()
    if args.data:
        logging.info(f'Loading dataset from {args.data}')
        df = utils.load_dataset(args.data)
    else:
        logging.info('No dataset path provided. Using default demo dataset.')
        df = utils.get_demo_dataset()

    report = train_model(df)
    print('Training complete.')
    print('Best model:', report['best_model'])
    print('Metrics:', report['metrics'])
    for name, metrics in report['candidates'].items():
        print(f'{name}: R2={metrics["r2"]:.4f}, MAE={metrics["mae"]:.2f}, RMSE={metrics["rmse"]:.2f}')


if __name__ == '__main__':
    main()
