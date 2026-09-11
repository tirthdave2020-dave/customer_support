import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.llm import classify_intent, get_relevant_cases, generate_response


GOLDEN_PATH = "eval/escalation_golden_set_30.csv"
OUTPUT_PATH = "eval/escalation_eval_results.csv"

df = pd.read_csv(GOLDEN_PATH)

# Load previous results if they exist
if os.path.exists(OUTPUT_PATH):
    results_df = pd.read_csv(OUTPUT_PATH)
else:
    results_df = pd.DataFrame(columns=[
        "case_id",
        "customer_message",
        "expected_action",
        "predicted_action",
        "intent",
        "reason"
    ])

for i, row in df.iterrows():

    case_id = row["case_id"]

    # Skip already completed examples
    if case_id in results_df["case_id"].values:
        continue

    print(f"Processing {i + 1}/{len(df)}")

    customer_message = row["customer_message"]

    try:
        # Step 1: classify intent
        intent = classify_intent(customer_message)

        # Step 2: retrieve historical cases
        results = get_relevant_cases(customer_message)

        # Step 3: generate reply + action
        response = generate_response(
            customer_message,
            intent,
            results
        )

        # Extract action and reason
        action = "unknown"
        reason = ""

        for line in response.splitlines():

            if line.strip().lower().startswith("action:"):
                action = line.split(":", 1)[1].strip().lower()

            if line.strip().lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        # Save completed example
        new_row = pd.DataFrame([{
            "case_id": case_id,
            "customer_message": customer_message,
            "expected_action": row["expected_action"],
            "predicted_action": action,
            "intent": intent,
            "reason": reason
        }])

        results_df = pd.concat(
            [results_df, new_row],
            ignore_index=True
        )

        # SAVE IMMEDIATELY
        results_df.to_csv(
            OUTPUT_PATH,
            index=False
        )

    except Exception as e:
        print(f"\nError: {e}")
        print("Stopping. Completed results have been saved.")
        break


# Evaluate only completed examples
if len(results_df) > 0:

    y_true = results_df["expected_action"]
    y_pred = results_df["predicted_action"]

    print("\n--- Escalation Evaluation ---")
    print(f"Completed: {len(results_df)}/{len(df)}")

    print(
        "Accuracy:",
        accuracy_score(y_true, y_pred)
    )

    print(
        "Macro F1:",
        f1_score(
            y_true,
            y_pred,
            average="macro"
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["auto_handle", "escalate"],
            zero_division=0
        )
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")