from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

def main():
    data_path = Path("data/raw/adult.data")
    models_path = Path("models")
    models_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        data_path,
        names=COLUMN_NAMES,
        na_values=" ?",
        skipinitialspace=True
    )

    df = df.dropna()

    X = df.drop("income", axis=1)
    y = df["income"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_estimators=100))
    ])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model.fit(X, y_encoded)

    joblib.dump(model, models_path / "model.pkl")
    joblib.dump(preprocessor, models_path / "scaler.pkl")
    joblib.dump(label_encoder, models_path / "encoders.pkl")

    print("Modelo y artefactos guardados en models/")

if __name__ == "__main__":
    main()