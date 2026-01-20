import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.features import extract_features

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    feats = [extract_features(u) for u in df["url"].astype(str).tolist()]
    return pd.DataFrame(feats)

def main():
    data_path = "data/raw/urls.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Missing data/raw/urls.csv with columns: url,label")

    df = pd.read_csv(data_path).dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)

    X = featurize(df)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/phishing_model.joblib")
    joblib.dump(list(X.columns), "models/feature_names.joblib")

    print("Saved model to models/phishing_model.joblib")

if __name__ == "__main__":
    main()
