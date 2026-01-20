import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

    model = joblib.load("models/phishing_model.joblib")
    preds = model.predict(X_test)

    os.makedirs("reports", exist_ok=True)

    # Save text report
    report = classification_report(y_test, preds, digits=3)
    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit (0)", "Phishing (1)"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix - Phishing URL Detector")
    plt.savefig("reports/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    print("Saved reports to reports/classification_report.txt and reports/confusion_matrix.png")

if __name__ == "__main__":
    main()
