# CSV Chat + ML Predictions

A Streamlit app that lets you chat with your CSV data and get AI-powered price predictions.

## Features

- Chat with CSV data using natural language
- ML price predictions for properties
- LM Studio integration for AI responses
- Interactive data exploration

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install streamlit pandas requests numpy scikit-learn joblib
   ```

2. **Setup LM Studio**:
   - Download [LM Studio](https://lmstudio.ai/)
   - Load a chat model (e.g., Mistral-7B-Instruct)
   - Start server on `localhost:1234`

3. **Configure paths** in `test-lm.py`:
   ```python
   CSV_FILE_PATH = "data/processed/merged_data.csv"
   ML_MODEL_PATH = "data/models/price_model.pkl"
   ```

4. **Run the app**:
   ```bash
   streamlit run test-lm.py
   ```

## Usage

**Ask questions about your data**:
- "Show me the first 10 rows"
- "What's the average price?"
- "Which properties have 3 bedrooms?"

**Get ML predictions**:
- Use the sidebar predictor with property features
- Ask "What features affect price most?"

The CSV and trained model are not included in this repository. Supply your own files and replace the existing machine-specific model path before running.

## Requirements

- Python 3.8+
- CSV file with your data
- Trained ML model (pickle file)
- LM Studio running locally

## Model Format

Your pickle file should contain:
```python
{
    'model': trained_model,
    'scaler': feature_scaler,
    'feature_names': ['accommodates', 'bedrooms', 'bathrooms', ...]
}
```

## Troubleshooting

- **LM Studio not connected**: Check it's running on `localhost:1234`
- **Model won't load**: Verify file path and pickle structure
- **CSV errors**: Check file path and format

That's it! Open `localhost:8501` and start chatting with your data.