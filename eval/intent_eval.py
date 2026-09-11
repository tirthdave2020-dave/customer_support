import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.llm import classify_intent


GOLDEN_PATH = "eval/golden_set_proposed_cleaned (2).csv"

df = pd.read_csv(GOLDEN_PATH)

# ADD THESE TWO PARTS HERE
df = df.dropna(subset=["proposed_label"])

y_true = df["proposed_label"].tolist()


predictions = []

for i, message in enumerate(df["customer_message"], 1):
    print(f"Processing {i}/{len(df)}")
    prediction = classify_intent(message)
    predictions.append(prediction)
accuracy = accuracy_score(y_true, predictions)
macro_f1 = f1_score(y_true, predictions, average="macro")

print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)

print("\nClassification Report:")
print(classification_report(y_true, predictions))