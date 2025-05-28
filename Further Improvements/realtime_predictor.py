import time
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from multi_stock_predictor import MultiStockPredictor
from sentiment_analyzer import MarketSentimentAnalyzer
import json
import os
from pathlib import Path
import logging

class RealtimePredictor:
    def __init__(
        self,
        tickers: List[str],
        update_interval: int = 15,  # minutes
        twitter_api_key: str = None,
        news_api_key: str = None
    ):
        self.tickers = tickers
        self.update_interval = update_interval
        self.predictor = MultiStockPredictor(tickers)
        self.sentiment_analyzers = {
            ticker: MarketSentimentAnalyzer(
                ticker,
                twitter_api_key=twitter_api_key,
                news_api_key=news_api_key
            )
            for ticker in tickers
        }
        
        # Setup logging
        self.setup_logging()
        
        # Create predictions directory
        self.predictions_dir = Path("predictions")
        self.predictions_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('realtime_predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_initial_models(self):
        """Train initial models for all stocks."""
        self.logger.info("Training initial models...")
        self.predictor.train_all_models()
        self.logger.info("Initial training completed")
    
    def update_predictions(self):
        """Update predictions for all stocks."""
        try:
            self.logger.info("Updating predictions...")
            
            predictions = {}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for ticker in self.tickers:
                try:
                    # Get sentiment data
                    sentiment_data = self.sentiment_analyzers[ticker].get_combined_sentiment(days_back=7)
                    
                    # Make prediction
                    stock_pred = self.predictor.predict(ticker)
                    
                    # Combine with sentiment
                    predictions[ticker] = {
                        'prediction': {
                            'lower': stock_pred['lower'].tolist(),
                            'median': stock_pred['median'].tolist(),
                            'upper': stock_pred['upper'].tolist()
                        },
                        'sentiment': {
                            'twitter_sentiment': sentiment_data['sentiment'].mean() if 'sentiment' in sentiment_data else None,
                            'news_sentiment': sentiment_data['news_sentiment'].mean() if 'news_sentiment' in sentiment_data else None
                        }
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error updating predictions for {ticker}: {str(e)}")
            
            # Save predictions
            prediction_file = self.predictions_dir / f"predictions_{timestamp.replace(' ', '_')}.json"
            with open(prediction_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'predictions': predictions
                }, f, indent=2)
            
            self.logger.info("Predictions updated successfully")
            
            # Clean up old prediction files (keep last 24 hours)
            self.cleanup_old_predictions()
            
        except Exception as e:
            self.logger.error(f"Error in update_predictions: {str(e)}")
    
    def cleanup_old_predictions(self):
        """Remove prediction files older than 24 hours."""
        try:
            current_time = datetime.now()
            for file in self.predictions_dir.glob("predictions_*.json"):
                file_time_str = file.stem.split('_', 1)[1]
                file_time = datetime.strptime(file_time_str, "%Y-%m-%d_%H:%M:%S")
                
                if current_time - file_time > timedelta(hours=24):
                    file.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning up old predictions: {str(e)}")
    
    def start(self):
        """Start the real-time prediction service."""
        self.logger.info("Starting real-time prediction service...")
        
        # Train initial models
        self.train_initial_models()
        
        # Schedule regular updates
        schedule.every(self.update_interval).minutes.do(self.update_predictions)
        
        # Run first update immediately
        self.update_predictions()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                self.logger.info("Stopping service...")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def get_latest_predictions(self) -> Dict:
        """Get the most recent predictions."""
        try:
            # Get most recent prediction file
            prediction_files = list(self.predictions_dir.glob("predictions_*.json"))
            if not prediction_files:
                return {}
            
            latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error getting latest predictions: {str(e)}")
            return {} 