
import pymongo
import pandas as pd
import matplotlib.pyplot as plt

def analyse_bourse():
    # Connexion à MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["bigdata_project"]  
    collection = db["stock_data"]  

    # Récupération des données
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)

    # Conversion de la colonne Date en datetime
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # Tri des données par date
    df = df.sort_values(by="Date")

    # Vérification des colonnes disponibles
    print("Colonnes disponibles :", df.columns)

    # Statistiques descriptives
    print("\nStatistiques descriptives :")
    print(df.describe())

    # Visualisation de l'évolution des prix (fermeture)
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], label="Prix de clôture", color="blue", linewidth=1)
    plt.fill_between(df["Date"], df["Close"], color="blue", alpha=0.3)
    plt.xlabel("Date")
    plt.ylabel("Prix de clôture")
    plt.title("Évolution du prix de clôture")
    plt.legend()
    plt.grid()
    plt.show()

    # Analyse des volumes
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Volume"], label="Volume", color="green")  
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.title("Évolution du volume des transactions")
    plt.legend()
    plt.grid()
    plt.show()

    # Variations en pourcentage
    df["PctChange"] = df["Close"].pct_change() * 100
    df["PctChange30"] = df["Close"].pct_change(periods=30) * 100  # sur 30 jours

    # Comparaison des performances par entreprise
    df_summary = df.groupby("symbol").agg(
        last_price=("Close", "last"),
        overall_change=("PctChange", "sum"),
        monthly_change=("PctChange30", "last")
    )
    print(df_summary)

    # Tendances avec moyennes mobiles
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    # Indicateur technique RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # Matrice de corrélation
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    correlation_matrix = df[numeric_columns].corr()
    print("\nMatrice de corrélation :")
    print(correlation_matrix)

    return df, df_summary

if __name__ == "__main__":
    df, df_summary = analyse_bourse()