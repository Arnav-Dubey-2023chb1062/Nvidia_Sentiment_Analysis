import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from transformer_model import StockDataPreparation, StockTFTModel
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
import torch
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

class MultiStockPredictor:
    def __init__(self, tickers: List[str], lookback_days: int = 60):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.models: Dict[str, StockTFTModel] = {}
        self.datasets: Dict[str, TimeSeriesDataSet] = {}
        
    def prepare_stock_data(self, ticker: str) -> pd.DataFrame:
        """Prepare data for a single stock."""
        data_prep = StockDataPreparation(ticker=ticker, lookback_days=self.lookback_days)
        return data_prep.get_stock_data(years=3)
    
    def prepare_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Prepare data for all stocks in parallel."""
        with ThreadPoolExecutor() as executor:
            future_to_ticker = {
                executor.submit(self.prepare_stock_data, ticker): ticker 
                for ticker in self.tickers
            }
            
            stock_data = {}
            for future in future_to_ticker:
                ticker = future_to_ticker[future]
                try:
                    stock_data[ticker] = future.result()
                except Exception as e:
                    print(f"Error preparing data for {ticker}: {str(e)}")
                    
        return stock_data
    
    def train_model(self, ticker: str, stock_data: pd.DataFrame) -> None:
        """Train model for a single stock."""
        try:
            # Create datasets
            training_data, validation_data, train_dataloader, val_dataloader = create_datasets(
                stock_data,
                ticker=ticker,
                max_prediction_length=5,
                max_encoder_length=60,
                batch_size=64
            )
            
            # Initialize and train model
            model = StockTFTModel(training_data)
            
            trainer = pl.Trainer(
                max_epochs=50,
                accelerator="cpu",
                devices=1,
                callbacks=[
                    pl.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        mode="min"
                    )
                ]
            )
            
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
            self.models[ticker] = model
            self.datasets[ticker] = training_data
            
        except Exception as e:
            print(f"Error training model for {ticker}: {str(e)}")
    
    def train_all_models(self) -> None:
        """Train models for all stocks."""
        stock_data = self.prepare_all_stocks()
        
        for ticker, data in stock_data.items():
            print(f"\nTraining model for {ticker}")
            self.train_model(ticker, data)
    
    def predict(self, ticker: str, days_ahead: int = 5) -> Dict[str, np.ndarray]:
        """Make predictions for a specific stock."""
        if ticker not in self.models:
            raise ValueError(f"No trained model found for {ticker}")
            
        model = self.models[ticker]
        dataset = self.datasets[ticker]
        
        # Get latest data
        stock = yf.Ticker(ticker)
        latest_data = stock.history(period=f"{self.lookback_days}d")
        
        # Prepare prediction data
        pred_data = dataset.get_prediction_sample(latest_data)
        
        # Make prediction
        raw_predictions = model(pred_data)
        
        # Extract quantile predictions
        predictions = {
            'lower': raw_predictions[:, :, 0].numpy(),  # 10th percentile
            'median': raw_predictions[:, :, 1].numpy(),  # 50th percentile
            'upper': raw_predictions[:, :, 2].numpy(),   # 90th percentile
        }
        
        return predictions
    
    def predict_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Make predictions for all stocks."""
        predictions = {}
        
        for ticker in self.tickers:
            try:
                predictions[ticker] = self.predict(ticker)
            except Exception as e:
                print(f"Error predicting for {ticker}: {str(e)}")
                
        return predictions
    
    def get_portfolio_prediction(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Get weighted prediction for a portfolio of stocks."""
        if weights is None:
            # Equal weights if not specified
            weights = {ticker: 1.0/len(self.tickers) for ticker in self.tickers}
            
        predictions = self.predict_all()
        
        # Initialize portfolio predictions
        portfolio_pred = {
            'lower': np.zeros(5),
            'median': np.zeros(5),
            'upper': np.zeros(5)
        }
        
        # Calculate weighted sum
        for ticker, pred in predictions.items():
            weight = weights.get(ticker, 0.0)
            for key in portfolio_pred:
                portfolio_pred[key] += weight * pred[key][0]
                
        return portfolio_pred 