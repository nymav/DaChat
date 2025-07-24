import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load processed data"""
    data_path = Path("data/processed/merged_data.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run process_data.py first.")
    
    data = pd.read_csv(data_path)
    logger.info(f"Loaded processed data: {len(data)} records")
    return data

def prepare_features(data):
    """Prepare features for training"""
    # Select relevant columns for price prediction
    feature_columns = [
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'review_count', 'availability_rate'
    ]
    
    # Add categorical features if they exist
    categorical_features = []
    if 'room_type' in data.columns:
        feature_columns.append('room_type')
        categorical_features.append('room_type')
    
    if 'property_type' in data.columns:
        feature_columns.append('property_type')
        categorical_features.append('property_type')
    
    if 'neighbourhood_cleansed' in data.columns:
        feature_columns.append('neighbourhood_cleansed')
        categorical_features.append('neighbourhood_cleansed')
    
    # Filter to available columns
    available_columns = [col for col in feature_columns if col in data.columns]
    
    # Create feature dataframe
    features = data[available_columns].copy()
    
    # Handle missing values
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = features[col].fillna('Unknown')
        else:
            features[col] = features[col].fillna(0)
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        if col in features.columns:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            label_encoders[col] = le
    
    logger.info(f"Prepared features: {features.shape[1]} columns")
    return features, label_encoders

def prepare_target(data):
    """Prepare target variable"""
    # Use price as target, with fallback to calendar_avg_price
    if 'price' in data.columns:
        target = data['price'].copy()
    elif 'calendar_avg_price' in data.columns:
        target = data['calendar_avg_price'].copy()
    else:
        raise ValueError("No price column found for target variable")
    
    # Remove outliers (prices > 95th percentile or < 5th percentile)
    q95 = target.quantile(0.95)
    q05 = target.quantile(0.05)
    target = target.clip(lower=q05, upper=q95)
    
    logger.info(f"Target variable prepared. Price range: ${target.min():.2f} - ${target.max():.2f}")
    return target

def train_model(X, y):
    """Train the price prediction model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Training Random Forest model...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    logger.info(f"Training Results:")
    logger.info(f"  Train RMSE: ${np.sqrt(train_mse):.2f}")
    logger.info(f"  Test RMSE: ${np.sqrt(test_mse):.2f}")
    logger.info(f"  Train R²: {train_r2:.3f}")
    logger.info(f"  Test R²: {test_r2:.3f}")
    
    return model, scaler, {
        'train_rmse': np.sqrt(train_mse),
        'test_rmse': np.sqrt(test_mse),
        'train_r2': train_r2,
        'test_r2': test_r2
    }

def save_model(model, scaler, label_encoders, feature_names, metrics):
    """Save the trained model and preprocessing objects"""
    models_dir = Path("data/models")
    models_dir.mkdir(exist_ok=True)
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'metrics': metrics
    }
    
    # Save model
    model_path = models_dir / "price_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = models_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        logger.info("Top 5 important features:")
        for idx, row in importance_df.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")

def main():
    """Main training pipeline"""
    try:
        # Load data
        data = load_processed_data()
        
        # Remove rows with missing target
        data = data.dropna(subset=['price'] if 'price' in data.columns else ['calendar_avg_price'])
        
        if len(data) < 100:
            raise ValueError("Insufficient data for training (need at least 100 records)")
        
        # Prepare features and target
        X, label_encoders = prepare_features(data)
        y = prepare_target(data)
        
        # Align X and y (remove rows where either is NaN)
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 50:
            raise ValueError("Insufficient valid data after cleaning")
        
        logger.info(f"Training with {len(X)} samples")
        
        # Train model
        model, scaler, metrics = train_model(X, y)
        
        # Save model
        save_model(model, scaler, label_encoders, list(X.columns), metrics)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
    main()