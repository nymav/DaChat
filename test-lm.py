import streamlit as st
import pandas as pd
import requests
import json
from typing import Dict, Any, List
import os
import logging
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Configuration
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
CSV_FILE_PATH = "data/processed/merged_data.csv"

ML_MODEL_PATH = "/Users/yarra/Desktop/German_Project/data/models/price_model.pkl" # UPDATE THIS TO YOUR ACTUAL MODEL PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
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

class CSVChatInterface:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.load_csv()
    
    def load_csv(self):
        """Load the CSV file and store basic info"""
        try:
            self.df = pd.read_csv(self.csv_path)
            st.sidebar.success(f"✅ Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading CSV: {str(e)}")
            self.df = None
    
    def get_csv_info(self) -> str:
        """Get basic information about the CSV structure"""
        if self.df is None:
            return "CSV file could not be loaded."
        
        # Get sample of data without overwhelming the context
        sample_data = self.df.head(2)  # Reduced to 2 rows
        
        info = f"""
CSV Dataset Overview:
- Total rows: {len(self.df):,}
- Total columns: {len(self.df.columns)}
- Column names: {', '.join(self.df.columns.tolist()[:10])}{'...' if len(self.df.columns) > 10 else ''}

Key Statistics:
- Numerical columns: {len(self.df.select_dtypes(include=['number']).columns)}
- Text columns: {len(self.df.select_dtypes(include=['object']).columns)}
- Missing data: {self.df.isnull().sum().sum()} total missing values

Sample data (first 2 rows):
{sample_data.to_string(max_cols=5)}
        """
        return info
    
    def get_detailed_context(self, query: str) -> str:
        """Get more detailed context based on the query"""
        if self.df is None:
            return "No data available."
        
        context = []
        query_lower = query.lower()
        
        # Add relevant context based on query
        if any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap']):
            if 'price' in self.df.columns:
                price_stats = self.df['price'].describe()
                context.append(f"Price statistics: min=${price_stats['min']:.2f}, max=${price_stats['max']:.2f}, avg=${price_stats['mean']:.2f}")
        
        if any(word in query_lower for word in ['location', 'neighborhood', 'area']):
            location_cols = [col for col in self.df.columns if 'location' in col.lower() or 'neighbourhood' in col.lower()]
            if location_cols:
                context.append(f"Location columns available: {', '.join(location_cols)}")
        
        if any(word in query_lower for word in ['review', 'rating', 'score']):
            review_cols = [col for col in self.df.columns if 'review' in col.lower() or 'rating' in col.lower()]
            if review_cols:
                context.append(f"Review/Rating columns: {', '.join(review_cols)}")
        
        return " | ".join(context) if context else "General dataset information available."
    
    def query_csv(self, query: str) -> str:
        """Execute queries on the CSV data"""
        if self.df is None:
            return "CSV data is not available."
        
        try:
            query_lower = query.lower()
            
            if "shape" in query_lower or "size" in query_lower:
                return f"The dataset has {self.df.shape[0]:,} rows and {self.df.shape[1]} columns."
            
            elif "columns" in query_lower:
                cols = self.df.columns.tolist()
                if len(cols) > 20:
                    return f"Dataset has {len(cols)} columns. First 20: {', '.join(cols[:20])}..."
                return f"Columns ({len(cols)}): {', '.join(cols)}"
            
            elif "describe" in query_lower or "summary" in query_lower:
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    return f"Numeric summary:\n{self.df[numeric_cols].describe().round(2).to_string()}"
                else:
                    return "No numeric columns found for statistical summary."
            
            elif "head" in query_lower or "first" in query_lower:
                n = 5
                if "10" in query_lower:
                    n = 10
                elif "3" in query_lower:
                    n = 3
                return f"First {n} rows:\n{self.df.head(n).to_string(max_cols=8)}"
            
            elif "null" in query_lower or "missing" in query_lower:
                missing = self.df.isnull().sum()
                missing_data = missing[missing > 0]
                if len(missing_data) == 0:
                    return "No missing values found in the dataset!"
                return f"Missing values per column:\n{missing_data.to_string()}"
            
            else:
                return "COMPLEX_QUERY"  # Signal that AI should handle this
                
        except Exception as e:
            return f"Error processing query: {str(e)}"

class LMStudioClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json", "Authorization": "Bearer lm-studio"}
        self.current_model = None
    
    def is_connected(self) -> bool:
        """Check if LM Studio is running"""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from LM Studio"""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = [model["id"] for model in models_data.get("data", [])]
                logger.info(f"Available models: {models}")
                return models
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
        return []
    
    def _is_raw_data_response(self, content: str) -> bool:
        """Check if response looks like raw data instead of natural language"""
        if not content or len(content.strip()) == 0:
            return True
            
        # Check for patterns that indicate raw data
        raw_indicators = [
            content.count(' ') < len(content) * 0.1,  # Very few spaces
            content.count('.') > len(content) * 0.1,  # Too many decimals
            len([c for c in content if c.isdigit()]) > len(content) * 0.5,  # Mostly numbers
            'id scrape_id host_id' in content.lower(),  # CSV column headers
            content.startswith(('11664.000000', '0.11664')),  # Common patterns
            len(content.split()) < 5 and content.count('.') > 3,  # Short with many decimals
        ]
        
        return any(raw_indicators)
    
    def chat_completion(self, messages: List[Dict], model: str = None, temperature: float = 0.7) -> str:
        """Send chat completion request to LM Studio with improved error handling"""
        try:
            # Clean up messages - ensure system message is properly formatted
            clean_messages = []
            for msg in messages:
                if msg["role"] in ["system", "user", "assistant"]:
                    clean_messages.append({
                        "role": msg["role"],
                        "content": str(msg["content"]).strip()
                    })
            
            # Ensure we have at least a user message
            if not any(msg["role"] == "user" for msg in clean_messages):
                return "Error: No user message found."
            
            payload = {
                "model": model or "mistral-7b-instruct-v0.1",
                "messages": clean_messages,
                "temperature": max(0.1, min(temperature, 1.0)),  # Clamp temperature
                "max_tokens": 1000,
                "top_p": 0.9,
                "stream": False,
                "stop": None,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1
            }
            
            logger.info(f"Sending request to LM Studio with {len(clean_messages)} messages")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60  # Increased timeout
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Check if we got raw data instead of a proper response
                if self._is_raw_data_response(content):
                    logger.warning("Received raw data response, trying alternative approach")
                    return "I'm having trouble generating a proper response. This might be a model configuration issue. Please ensure LM Studio is in 'Chat' mode and using an instruct model like Mistral-7B-Instruct."
                
                return content
            else:
                error_msg = f"LM Studio API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except requests.exceptions.Timeout:
            return "Request timed out. The model might be processing a complex query."
        except requests.exceptions.ConnectionError:
            return "Connection error. Please make sure LM Studio is running on localhost:1234."
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return error_msg

class PredictionModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None  # Will be determined from your model
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model"""
        try:
            model_data = joblib.load(self.model_path)
            
            # Handle different model file structures
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_columns = model_data.get('features')
                self.scaler = model_data.get('scaler', None)
            else:
                self.model = model_data
                # Default features - adjust based on your model
                self.feature_columns = [
                    'accommodates', 'bedrooms', 'bathrooms', 'latitude', 'longitude',
                    'host_is_superhost', 'number_of_reviews', 'review_scores_rating',
                    'availability_365', 'minimum_nights'
                ]
            
            st.sidebar.success("✅ ML Model loaded successfully")
            logger.info("ML model loaded successfully")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading ML model: {str(e)}")
            logger.error(f"Failed to load ML model: {e}")
            self.model = None
    
    def prepare_input_features(self, **kwargs) -> np.ndarray:
        """Prepare input features for prediction"""
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined")
        
        features = []
        for col in self.feature_columns:
            if col in kwargs:
                features.append(kwargs[col])
            else:
                # Use default values for missing features
                default_values = {
                    'accommodates': 2, 'bedrooms': 1, 'bathrooms': 1,
                    'latitude': 47.5, 'longitude': 19.05,  # Budapest center
                    'host_is_superhost': 0, 'number_of_reviews': 10,
                    'review_scores_rating': 4.5, 'availability_365': 200,
                    'minimum_nights': 2
                }
                features.append(default_values.get(col, 0))
        
        return np.array(features).reshape(1, -1)
    
    def predict_price(self, **property_features) -> float:
        """Predict property price"""
        if self.model is None:
            return None
        
        try:
            features = self.prepare_input_features(**property_features)
            
            # Apply scaling if scaler exists
            if hasattr(self, 'scaler') and self.scaler is not None:
                features = self.scaler.transform(features)
            
            prediction = self.model.predict(features)[0]
            return max(0, prediction)  # Ensure non-negative price
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def get_feature_importance(self) -> dict:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {}

def handle_prediction_query(query: str, ml_service: PredictionModel, csv_chat: CSVChatInterface) -> str:
    """Handle ML prediction queries"""
    if ml_service.model is None:
        return "❌ ML model is not available for predictions."
    
    query_lower = query.lower()
    
    # Price prediction queries
    if any(word in query_lower for word in [' price', 'price ', 'estimate price', 'what price']):
        return "🔮 **Price Prediction Available!** \n\nPlease use the 'Price Predictor' in the sidebar to get a price estimate for a property based on its features."
    
    # Feature importance
    elif any(word in query_lower for word in ['feature importance', 'important features', 'what affects price']):
        importance = ml_service.get_feature_importance()
        if importance:
            response = "📊 **Most Important Features for Price Prediction:**\n\n"
            for i, (feature, score) in enumerate(list(importance.items())[:5]):
                response += f"{i+1}. **{feature.replace('_', ' ').title()}**: {score:.3f}\n"
            return response
        else:
            return "Feature importance information is not available for this model."
    
    # Model info
    elif any(word in query_lower for word in ['model info', 'model type', 'algorithm', 'ml model']):
        model_type = type(ml_service.model).__name__
        feature_count = len(ml_service.feature_columns) if ml_service.feature_columns else "Unknown"
        return f"🤖 **ML Model Information:**\n- Algorithm: {model_type}\n- Features used: {feature_count}\n- Prediction type: Price Prediction"
    
    return None

def main():
    st.set_page_config(
        page_title="CSV Chat with LM Studio + ML",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 CSV Data Chat Interface + ML Predictions")
    st.markdown("Chat with your data and get ML-powered predictions using LM Studio")
    
    # Initialize components
    csv_chat = CSVChatInterface(CSV_FILE_PATH)
    lm_client = LMStudioClient(LM_STUDIO_BASE_URL)
    
    # Initialize ML service
    ml_service = PredictionModel(ML_MODEL_PATH)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Info")
        
        if csv_chat.df is not None:
            st.metric("Rows", f"{len(csv_chat.df):,}")
            st.metric("Columns", len(csv_chat.df.columns))
            
            with st.expander("Column Details"):
                for i, col in enumerate(csv_chat.df.columns):
                    if i < 20:  # Limit display
                        st.text(f"• {col}")
                    elif i == 20:
                        st.text(f"... and {len(csv_chat.df.columns) - 20} more")
                        break
        
        # ML Prediction Section
        st.header("🔮 Price Predictor")
        
        if ml_service.model is not None:
            with st.expander("Predict Property Price", expanded=False):
                # Input fields for prediction
                accommodates = st.number_input("Accommodates", 1, 20, 2)
                bedrooms = st.number_input("Bedrooms", 1, 10, 1)
                bathrooms = st.number_input("Bathrooms", 1, 10, 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    latitude = st.number_input("Latitude", 47.0, 48.0, 47.5)
                with col2:
                    longitude = st.number_input("Longitude", 18.5, 19.5, 19.05)
                
                is_superhost = st.checkbox("Superhost")
                num_reviews = st.number_input("Number of Reviews", 0, 1000, 10)
                rating = st.slider("Rating", 1.0, 5.0, 4.5, 0.1)
                availability = st.slider("Availability (days/year)", 0, 365, 200)
                min_nights = st.number_input("Minimum Nights", 1, 30, 2)
                
                if st.button("🔮 Predict Price"):
                    prediction = ml_service.predict_price(
                        accommodates=accommodates,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        latitude=latitude,
                        longitude=longitude,
                        host_is_superhost=1 if is_superhost else 0,
                        number_of_reviews=num_reviews,
                        review_scores_rating=rating,
                        availability_365=availability,
                        minimum_nights=min_nights
                    )
                    
                    if prediction is not None:
                        st.success(f"💰 Predicted Price: ${prediction:,.2f}")
                        
                        # Add to chat
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        prediction_msg = f"Based on the property features you provided, the predicted price is ${prediction:,.2f}"
                        st.session_state.messages.append({"role": "assistant", "content": prediction_msg})
                    else:
                        st.error("❌ Prediction failed")
        else:
            st.warning("⚠️ ML model not loaded")
        
        st.header("🤖 LM Studio Status")
        is_connected = lm_client.is_connected()
        
        if is_connected:
            st.success("✅ Connected")
            models = lm_client.get_models()
            if models:
                # Default to mistral model if available
                default_idx = 0
                for i, model in enumerate(models):
                    if "mistral" in model.lower():
                        default_idx = i
                        break
                
                selected_model = st.selectbox("Select Model", models, index=default_idx)
                lm_client.current_model = selected_model
            else:
                selected_model = None
                st.warning("⚠️ No models loaded in LM Studio")
                st.info("Load a model in LM Studio first")
        else:
            st.error("❌ Not connected")
            st.info("Make sure LM Studio is running on localhost:1234")
            selected_model = None
        
        # Temperature setting
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.json({
                "Connected": is_connected,
                "Selected Model": selected_model,
                "Temperature": temperature,
                "CSV Loaded": csv_chat.df is not None,
                "ML Model Loaded": ml_service.model is not None
            })
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't display system messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your data or request ML predictions..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                # First check for ML prediction queries
                ml_response = handle_prediction_query(prompt, ml_service, csv_chat)
                
                if ml_response:
                    response = ml_response
                else:
                    # Try simple CSV queries
                    csv_response = csv_chat.query_csv(prompt)
                    
                    if csv_response != "COMPLEX_QUERY":
                        response = csv_response
                    else:
                        # Use LM Studio for complex queries
                        if is_connected and selected_model:
                            csv_context = csv_chat.get_csv_info()
                            detailed_context = csv_chat.get_detailed_context(prompt)
                            
                            system_message = {
                                "role": "system",
                                "content": f"""You are a helpful data analyst assistant with ML prediction capabilities. 

{csv_context}

Additional context: {detailed_context}

You can also help with:
- Price predictions using machine learning
- Feature importance analysis
- Data insights and patterns

Provide clear, helpful insights about the data."""
                            }
                            
                            messages_for_ai = [system_message] + st.session_state.messages[-10:]
                            response = lm_client.chat_completion(messages_for_ai, model=selected_model, temperature=temperature)
                        else:
                            response = "🚫 LM Studio is not connected or no model is selected."
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("**Quick Actions:**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("📊 Data Summary"):
            st.session_state.messages.append({"role": "user", "content": "Show me a summary of the data"})
            st.rerun()
    
    with col2:
        if st.button("📋 Sample Data"):
            st.session_state.messages.append({"role": "user", "content": "Show me the first 5 rows"})
            st.rerun()
    
    with col3:
        if st.button("📈 Data Types"):
            st.session_state.messages.append({"role": "user", "content": "What are the column types and data structure?"})
            st.rerun()
    
    with col4:
        if st.button("🔍 Missing Data"):
            st.session_state.messages.append({"role": "user", "content": "Check for missing values"})
            st.rerun()
    
    with col5:
        if st.button("🔮 ML Features"):
            st.session_state.messages.append({"role": "user", "content": "What features are most important for price prediction?"})
            st.rerun()
    
    with col6:
        if st.button("🤖 Model Info"):
            st.session_state.messages.append({"role": "user", "content": "Tell me about the ML model"})
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer with tips
    if len(st.session_state.messages) == 0:
        st.markdown("---")
        st.markdown("""
        **💡 Tips for better results:**
        - Ask specific questions about your data
        - Use natural language (e.g., "What are the most expensive properties?")
        - Request ML price predictions using the sidebar predictor
        - Ask about feature importance and model performance
        - Request visualizations or specific analyses
        """)

if __name__ == "__main__":
  main()