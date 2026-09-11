import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ==========================================
# 1. Load golden dataset
# ==========================================

df = pd.read_csv("eval/golden_set (2).csv")

# Remove the one missing label
df = df.dropna(subset=["label"])

print(f"Total labeled examples: {len(df)}")


# ==========================================
# 2. Split into training and test data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["customer_message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")


# ==========================================
# 3. Convert text into TF-IDF features
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 4. Train Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 5. Predict
# ==========================================

predictions = model.predict(X_test_tfidf)


# ==========================================
# 6. Evaluate
# ==========================================

accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print("\n==========================================")
print("Baseline 2: TF-IDF + Logistic Regression")
print("==========================================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)