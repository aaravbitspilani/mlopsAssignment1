import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.data_loader import load_data

def train_model():
    df = load_data()
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for model_cls in [LinearRegression, DecisionTreeRegressor]:
        with mlflow.start_run():
            model = model_cls()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mlflow.log_param("model", model_cls.__name__)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train_model()
