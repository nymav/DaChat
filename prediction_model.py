import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PredictionModel:
    """Wrapper for the trained price prediction model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.metrics = {}
        self.is_loaded = False
        
        if model_path is None:
            model_path = Path("data/models/price_model.pkl")
        
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model from file"""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found at {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.label_encoders = model_package.get('label_encoders', {})
            self.feature_names = model_package.get('feature_names', [])
            self.metrics = model_package.get('metrics', {})
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully. Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def _prepare_features(self, input_data):
        """Prepare input features for prediction"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Convert to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input must be a dictionary or DataFrame")
        
        # Create feature dataframe with correct columns
        features = pd.DataFrame(columns=self.feature_names)
        
        # Fill in available features
        for col in self.feature_names:
            if col in input_data.columns:
                features[col] = input_data[col]
            else:
                # Set default values for missing features
                features[col] = 0 if col in ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                                           'review_count', 'availability_rate'] else 'Unknown'
        
        # Handle missing values
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = features[col].fillna('Unknown')
            else:
                features[col] = features[col].fillna(0)
        
        # Apply label encoding for categorical features
        for col, encoder in self.label_encoders.items():
            if col in features.columns:
                # Handle unseen categories
                def safe_transform(x):
                    try:
                        return encoder.transform([str(x)])[0]
                    except ValueError:
                        # Return encoding for 'Unknown' or first class
                        if 'Unknown' in encoder.classes_:
                            return encoder.transform(['Unknown'])[0]
                        else:
                            return encoder.transform([encoder.classes_[0]])[0]
                
                features[col] = features[col].apply(safe_transform)
        
        return features
    
    def predict(self, input_data):
        """Make price predictions"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Please train a model first.")
        
        try:
            # Prepare features
            features = self._prepare_features(input_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            # Ensure positive predictions
            predictions = np.maximum(predictions, 0)
            
            return predictions.tolist() if len(predictions) > 1 else predictions[0]
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Failed to make prediction: {str(e)}")
    
    def predict_with_confidence(self, input_data, confidence_interval=0.95):
        """Make predictions with confidence intervals (if supported by model)"""
        predictions = self.predict(input_data)
        
        # For Random Forest, we can use the predictions from individual trees
        # to estimate uncertainty
        if hasattr(self.model, 'estimators_'):
            try:
                features = self._prepare_features(input_data)
                features_scaled = self.scaler.transform(features)
                
                # Get predictions from all trees
                tree_predictions = np.array([
                    tree.predict(features_scaled) for tree in self.model.estimators_
                ])
                
                # Calculate confidence intervals
                alpha = 1 - confidence_interval
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bounds = np.percentile(tree_predictions, lower_percentile, axis=0)
                upper_bounds = np.percentile(tree_predictions, upper_percentile, axis=0)
                
                if isinstance(predictions, list):
                    return {
                        'predictions': predictions,
                        'lower_bounds': lower_bounds.tolist(),
                        'upper_bounds': upper_bounds.tolist()
                    }
                else:
                    return {
                        'prediction': predictions,
                        'lower_bound': lower_bounds[0],
                        'upper_bound': upper_bounds[0]
                    }
                    
            except Exception as e:
                logger.warning(f"Could not calculate confidence intervals: {e}")
        
        # Fallback: return prediction without confidence intervals
        if isinstance(predictions, list):
            return {'predictions': predictions}
        else:
            return {'prediction': predictions}
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if not self.is_loaded:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {
                name: importance 
                for name, importance in zip(self.feature_names, self.model.feature_importances_)
            }
            return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "features": self.feature_names,
            "num_features": len(self.feature_names),
            "metrics": self.metrics
        }
        
        return info
    
    def validate_input(self, input_data):
        """Validate input data structure"""
        if not isinstance(input_data, (dict, pd.DataFrame)):
            return False, "Input must be a dictionary or DataFrame"
        
        # Convert to dict for easier checking
        if isinstance(input_data, pd.DataFrame):
            if len(input_data) == 0:
                return False, "Empty DataFrame provided"
            input_dict = input_data.iloc[0].to_dict()
        else:
            input_dict = input_data
        
        # Check for required numeric features
        required_numeric = ['accommodates']
        for feature in required_numeric:
            if feature in input_dict:
                try:
                    float(input_dict[feature])
                except (ValueError, TypeError):
                    return False, f"Feature '{feature}' must be numeric"
        
        return True, "Valid input"