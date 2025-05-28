# Enhanced Stock Price Prediction Model using Temporal Fusion Transformer

This project implements an advanced stock price prediction model using PyTorch Lightning and the Temporal Fusion Transformer (TFT) architecture, specifically optimized for NVIDIA (NVDA) stock predictions with multi-stock support and real-time sentiment analysis.

## Key Improvements Over Previous Models

1. **Enhanced Feature Engineering**
   - Multiple timeframe technical indicators (5, 10, 20, 50 days)
   - Advanced price indicators:
     * Multiple Moving Averages with standard deviations
     * RSI with different windows (14, 28 days)
     * Enhanced MACD with signal line and histogram
     * Bollinger Bands (20 and 50-day)
     * Price momentum indicators
   - Volume-based features:
     * Volume moving averages
     * Volume ratios
     * Price-volume relationships
   - Market timing features:
     * Session indicators (month/quarter/year start/end)
     * Temporal features (day, week, month patterns)
   - Sentiment features:
     * Twitter sentiment analysis
     * News sentiment processing
     * Weighted engagement scores
     * Sentiment momentum indicators

2. **Improved Model Architecture**
   - Temporal Fusion Transformer implementation:
     * Variable selection networks
     * Gated Residual Networks (GRNs)
     * Multi-head attention mechanisms
     * Temporal self-attention
     * Quantile regression layers
   - Architecture enhancements:
     * Larger hidden size (64 vs 32)
     * More attention heads (4 vs 2)
     * Increased hidden continuous size (32 vs 16)
     * Optimized dropout (0.2) for better regularization
     * Reduced learning rate (0.001) for more stable training

3. **Better Training Configuration**
   - Data improvements:
     * Extended historical data (3 years vs 2 years)
     * Larger batch size (64 vs 32)
     * More training epochs (50 vs 30)
     * Increased early stopping patience (10 vs 5)
     * Doubled training batches (100 vs 50)
   - Training optimizations:
     * Gradient clipping
     * Learning rate scheduling
     * Batch normalization
     * Early stopping mechanisms

4. **Enhanced Data Processing**
   - Advanced preprocessing:
     * Proper handling of categorical variables
     * Advanced data normalization techniques
     * Better missing data handling
     * Improved data validation checks
   - Real-time processing:
     * Continuous data updates
     * Parallel processing
     * Memory optimization
     * Error handling

## Advanced Features

1. **Multi-Stock Support**
   - Parallel stock data processing
   - Individual model training per stock
   - Portfolio-level predictions
   - Cross-asset feature analysis
   - Resource-optimized implementation

2. **Real-Time Predictions**
   - Continuous model updates
   - Regular prediction refreshes
   - Historical prediction storage
   - Automated cleanup systems
   - Comprehensive logging

3. **Sentiment Analysis Integration**
   - Real-time Twitter sentiment analysis
   - News sentiment processing
   - Engagement-weighted scoring
   - Sentiment technical indicators
   - Combined sentiment metrics

4. **Risk Management**
   - Uncertainty quantification
   - Confidence intervals
   - Multiple prediction horizons
   - Portfolio optimization
   - Risk-aware predictions

## Technical Architecture

### Deep Learning Components
- PyTorch Lightning framework
- Temporal Fusion Transformer
- Multi-head attention mechanisms
- Quantile regression
- Transfer learning capabilities

### Natural Language Processing
- Sentiment analysis engines
- Text preprocessing
- Entity recognition
- Document-term matrices
- Feature extraction

### Time Series Analysis
- Multiple time horizons
- Technical indicators
- Seasonal patterns
- Trend analysis
- Volatility modeling

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- PyTorch Forecasting 1.0+
- Additional dependencies:
  * tweepy>=4.12.0 (Twitter API)
  * textblob>=0.17.1 (NLP)
  * newsapi-python>=0.2.7 (News API)
  * schedule>=1.2.0 (Task scheduling)
  * concurrent-futures>=3.0.5 (Parallel processing)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
# Run the model with default settings
python transformer_model.py
```

### Advanced Usage
```python
# Multi-stock prediction with sentiment analysis
from realtime_predictor import RealtimePredictor

predictor = RealtimePredictor(
    tickers=["NVDA", "AMD", "INTC"],
    update_interval=15,
    twitter_api_key="your_key",
    news_api_key="your_key"
)
predictor.start()
```

## Model Performance

1. **Prediction Accuracy**
   - Improved mean absolute error
   - Better RMSE scores
   - Enhanced directional accuracy
   - Reliable uncertainty estimates

2. **Training Efficiency**
   - Faster convergence
   - Optimized resource usage
   - Reduced overfitting
   - Stable training process

3. **Real-World Performance**
   - Accurate price predictions
   - Better volatility handling
   - Improved risk assessment
   - Reliable uncertainty bounds

## Future Improvements

1. **Model Enhancements**
   - Advanced attention mechanisms
   - Dynamic feature selection
   - Adaptive learning rates
   - Transfer learning integration

2. **Feature Engineering**
   - Alternative data sources
   - Market microstructure
   - Order book data
   - Cross-market indicators

3. **System Improvements**
   - GPU acceleration
   - Distributed training
   - Real-time optimization
   - Advanced caching

4. **Risk Management**
   - Dynamic risk adjustment
   - Portfolio optimization
   - Automated trading signals
   - Risk-aware predictions

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License - feel free to use this code for any purpose.

## Acknowledgments

- PyTorch Forecasting team for the TFT implementation
- PyTorch Lightning team for the training framework
- yfinance for providing market data access
- Twitter and NewsAPI for sentiment data access

## Important Notes

- API keys required for full functionality
- Regular system maintenance recommended
- Performance varies by market conditions
- Not financial advice - use responsibly
- Consider market risks

