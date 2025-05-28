# NVIDIA Stock Price Prediction Project

## Overview
This project implements a machine learning model to predict NVIDIA (NVDA) stock prices using XGBoost, incorporating both technical indicators and sentiment analysis. The project consists of two main components:
1. A Python implementation of the prediction model (`Initial.py`)
2. Technical documentation of improvements and features (`model_improvements.txt`)
3. A Python implementation of the final prediction model (`Final.py`)

## Project Structure
```
├── Initial.py              # Initial implementation file
├── Finalal.py              # Final implementation file
├── model_improvements.txt    # Technical documentation
└── README.md                # Project documentation
```

## Requirements
- Python 3.x
- Required packages:
  ```
  yfinance
  pandas
  numpy
  nltk
  vaderSentiment
  matplotlib
  requests
  xgboost
  optuna
  ```

## Implementation Details (Final.py)

### Core Components

1. **Data Collection (`DataCollector` class)**
   - Fetches NVIDIA stock data using yfinance
   - Implements synthetic sentiment analysis
   - Handles data preprocessing and cleaning

2. **Feature Engineering (`FeatureEngineering` class)**
   - Technical indicators (MA5, MA20, RSI, MACD)
   - Lag features for price and volume
   - Sentiment score integration
   - Volatility and return calculations

3. **Stock Prediction (`StockPredictor` class)**
   - XGBoost-based prediction model
   - Rolling window forecasting
   - Performance metrics calculation

### Key Features
- 2-year historical data analysis
- Daily price prediction
- Synthetic sentiment analysis
- Multiple technical indicators
- Robust error handling
- Performance visualization

## Technical Improvements (model_improvements.txt)

### Major Enhancements

1. **Data Collection Improvements**
   - Extended historical data (2 years)
   - Enhanced synthetic sentiment analysis
   - Improved data preprocessing

2. **Technical Analysis**
   - Multiple moving averages
   - RSI with improved handling
   - MACD implementation

3. **Feature Engineering**
   - Optimized lag features
   - Additional technical indicators
   - Better data processing

4. **Model Architecture**
   - Simplified prediction horizon
   - Optimized XGBoost parameters
   - Improved validation methods

## Usage

1. **Setup**
   ```bash
   # Install required packages
   pip install yfinance pandas numpy nltk vaderSentiment matplotlib requests xgboost optuna
   ```

2. **Run the Model**
   ```python
   python Final.py
   ```

3. **Output**
   - Stock price predictions
   - Performance metrics (RMSE, MAE, R²)
   - Visualization of actual vs predicted prices

## Model Performance

The model provides:
- Short-term (1-day) price predictions
- Technical indicator-based analysis
- Sentiment integration
- Performance metrics visualization

### Metrics Used
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)

## Future Improvements

1. **Data Enhancement**
   - Real sentiment data integration
   - Additional technical indicators
   - Market sector analysis

2. **Model Optimization**
   - Advanced feature engineering
   - Hyperparameter tuning
   - Alternative model architectures

3. **Additional Features**
   - Volume-weighted indicators
   - Economic indicators
   - Market correlation analysis

## Development Process

### Initial Implementation
1. Basic model setup with XGBoost
2. Technical indicator implementation
3. Basic sentiment analysis
4. Initial error handling

### Improvements
1. Extended historical data
2. Enhanced feature engineering
3. Improved error handling
4. Better visualization
5. Comprehensive documentation

## Notes
- The model uses synthetic sentiment data due to API limitations
- Focuses on practical implementation over complex optimization
- Balances model complexity with reliability
- Suitable for educational and research purposes

## Disclaimer
This project is for educational purposes only. The predictions should not be used as financial advice. Always conduct thorough research and consult financial professionals before making investment decisions.

