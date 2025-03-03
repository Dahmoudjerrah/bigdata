import yfinance as yf
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

# Connexion à MongoDB sur Windows
client = MongoClient("mongodb://localhost:27017/")
db = client["bigdata_project"]
collection = db["stock_data"]

# Liste des entreprises du CAC 40 et autres grandes entreprises
stocks = ["GOOGL", "AAPL", "MSFT", "AMZN", "META", "BNP.PA", "AI.PA", "OR.PA"]

def collect_and_store():
    for stock in stocks:
        try:
            # Récupération des données
            ticker = yf.Ticker(stock)
            data = ticker.history(period="5y")  # 5 ans d'historique
            
            # Préparation des données pour MongoDB
            records = data.reset_index().to_dict("records")
            
            for record in records:
                # Conversion des types pour MongoDB
                record["symbol"] = stock
                if isinstance(record["Date"], pd.Timestamp):
                    record["Date"] = record["Date"].isoformat()
                
                # Insertion dans MongoDB
                collection.insert_one(record)
                
            print(f"Données stockées pour {stock}")
            
        except Exception as e:
            print(f"Erreur pour {stock}: {str(e)}")

if __name__ == "__main__":
    collect_and_store()