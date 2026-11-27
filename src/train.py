from pathlib import Path

import joblib

import pandas as pd

from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets"
MODEL_PATH = BASE_DIR / "ruqum" / "models" / "ruqum.joblib"


def train():
    train_df = pd.read_csv(filepath_or_buffer=DATA_DIR / "train.csv")
    test_df = pd.read_csv(filepath_or_buffer=DATA_DIR / "test.csv")

    model = make_pipeline(
        TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            sublinear_tf=True
        ),
        LogisticRegression(
            max_iter=1000,
            C=2.0
        )
    )

    model.fit(
        train_df['text'],
        train_df['label']
    )

    test_predictions = model.predict(test_df['text'])

    print(f"TF-IDF Accuracy: {accuracy_score(test_df['label'], test_predictions):.4f}\n")
    print(classification_report(test_df['label'], test_predictions, digits=4))

    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    train()
