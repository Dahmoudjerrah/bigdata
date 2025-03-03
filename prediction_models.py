
from analyse_bourse import analyse_bourse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np

def train_models():
    df, _ = analyse_bourse()

    features = ["Open", "High", "Low", "Volume", "RSI", "SMA50", "SMA200"]
    target = "Close"

    X = df[features]  
    y = df[target]

    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Régression linéaire
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print(f"Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred))}")
    print(f"Linear Regression R2: {r2_score(y_test, lr_pred)}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100)  
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test) 
    print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred))}")  
    print(f"Random Forest R2: {r2_score(y_test, rf_pred)}")

    return y_test, lr_pred

if __name__ == "__main__":
    y_test, lr_pred = train_models()