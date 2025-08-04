import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_loader import load_data
import joblib


def train_model():
    df = load_data()
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for model_cls in [LinearRegression, DecisionTreeRegressor]:
        mlflow.set_experiment("california_housing_experiment")
        with mlflow.start_run():
            model = model_cls()
            model.fit(X_train, y_train)
            # Save the selected model
            joblib.dump(model, "models/best_model.pkl")
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mlflow.log_param("model", model_cls.__name__)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    train_model()
