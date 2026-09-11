import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load golden evaluation set
df = pd.read_csv("eval/golden_set (2).csv")

# Ignore the one unlabeled example for evaluation
df = df.dropna(subset=["label"])

# Majority-class prediction
majority_class = df["label"].value_counts().idxmax()
predictions = [majority_class] * len(df)

# Metrics
accuracy = accuracy_score(df["label"], predictions)
macro_f1 = f1_score(df["label"], predictions, average="macro", zero_division=0)

print("Baseline 1: Majority Class")
print("--------------------------")
print(f"Majority intent: {majority_class}")
print(f"Evaluation examples: {len(df)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    df["label"],
    predictions,
    zero_division=0
))