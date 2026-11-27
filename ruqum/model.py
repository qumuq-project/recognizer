import re
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ruqum.joblib"


class RuQumRecognizer:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        self.model = joblib.load(model_path)

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^а-яё]+", " ", text.lower()).strip()

    def predict(self, text: str) -> tuple[str, float]:
        cleaned = self._clean(text)
        if not cleaned:
            return "unknown", 0.0

        probs = self.model.predict_proba([cleaned])[0]
        best_idx = probs.argmax()

        lang = str(self.model.classes_[best_idx])
        confidence = float(probs[best_idx])
        return lang, round(confidence, 4)


if __name__ == "__main__":
    recognizer = RuQumRecognizer()

    test_phrases = [
        "Ассаламу алейкум, нечик сиз?",
        "Сегодня в Махачкале отличная погода.",
        "12345 !!!",
    ]

    for phrase in test_phrases:
        lang, conf = recognizer.predict(phrase)
        print(f"[{lang} | {conf:.4f}] -> {phrase}")
