"""
Training script for the Carbon Emission Calculator.
This script runs the complete training pipeline.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve

from data.data_loader import FlightDataLoader
from data.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from utils.visualization import ModelVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete training pipeline."""
    try:
        # 1. Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_loader = FlightDataLoader('airline_routes.json')
        df = data_loader.preprocess_data()
        logger.info(f"Data shape: {df.shape}")
        
        # 2. Feature engineering
        logger.info("Performing feature engineering...")
        feature_engineer = FeatureEngineer(df)
        df_processed = feature_engineer.create_features()
        logger.info(f"Processed data shape: {df_processed.shape}")
        
        # 3. Prepare data for training
        # Get feature names (excluding target and non-feature columns)
        feature_names = feature_engineer.get_feature_names()
        logger.info(f"Number of features: {len(feature_names)}")
        logger.info(f"Feature names: {feature_names}")
        
        X = df_processed[feature_names]
        y = df_processed['carbon_emission']
        
        # 4. Train model
        logger.info("Training model...")
        trainer = ModelTrainer()
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        
        # Train model with hyperparameter tuning
        logger.info("Starting model training with hyperparameter tuning...")
        model = trainer.train_model(X_train, y_train, tune_hyperparameters=True)
        
        # 5. Evaluate model
        logger.info("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"Model metrics: {metrics}")
        
        # 6. Generate predictions for visualization
        y_pred = model.predict(X_test)
        
        # 7. Create visualizations
        logger.info("Creating visualizations...")
        visualizer = ModelVisualizer()
        
        # Plot feature importance
        feature_importance = model.get_feature_importance()
        visualizer.plot_feature_importance(feature_importance)
        logger.info("Feature importance plot created")
        
        # Plot predictions vs actual
        visualizer.plot_prediction_vs_actual(y_test, y_pred)
        logger.info("Prediction vs actual plot created")
        
        # Plot error distribution
        visualizer.plot_error_distribution(y_test, y_pred)
        logger.info("Error distribution plot created")
        
        # Plot learning curves
        logger.info("Calculating learning curves...")
        train_sizes, train_scores, test_scores = learning_curve(
            model.model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        visualizer.plot_learning_curves(train_sizes, -train_scores, -test_scores)
        logger.info("Learning curves plot created")
        
        # 8. Save results
        logger.info("Saving results...")
        trainer.save_training_results()
        
        # Save feature importance
        feature_importance.to_csv('training_results/feature_importance.csv')
        
        logger.info("Training pipeline completed successfully!")
        logger.info("Visualizations saved in 'training_results' directory")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 