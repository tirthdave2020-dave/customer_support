import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.llm import classify_intent, get_relevant_cases, generate_response


GOLDEN_PATH = "eval/golden_set_proposed_cleaned (2).csv"
OUTPUT_PATH = "eval/reply_eval_results.csv"

df = pd.read_csv(GOLDEN_PATH)
df = df.dropna(subset=["proposed_label"])

# Use only 20 examples
df = df.head(5)

# Resume if results already exist
if os.path.exists(OUTPUT_PATH):
    results_df = pd.read_csv(OUTPUT_PATH)
else:
    results_df = pd.DataFrame(columns=[
        "case_id",
        "customer_message",
        "intent",
        "evidence",
        "reply",
        "action",
        "reason"
    ])


for i, row in df.iterrows():

    case_id = row["case_id"]

    if case_id in results_df["case_id"].values:
        continue

    print(f"Processing {i + 1}/{len(df)}")

    customer_message = row["customer_message"]

    try:
        # 1. Intent
        intent = classify_intent(customer_message)

        # 2. Retrieval
        retrieved = get_relevant_cases(customer_message)

        # 3. Generate response
        response = generate_response(
            customer_message,
            intent,
            retrieved
        )

        # Save retrieved evidence as text
        evidence = "\n\n".join(
            [
                f"Customer: {' '.join(r['case']['customer_messages'])}\n"
                f"AmazonHelp: {' '.join(r['case']['amazon_replies'])}\n"
                f"Similarity: {r['score']:.4f}"
                for r in retrieved
            ]
        )

        # Extract action
        action = "unknown"
        reason = ""

        for line in response.splitlines():
            if line.strip().lower().startswith("action:"):
                action = line.split(":", 1)[1].strip()

            if line.strip().lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        new_row = pd.DataFrame([{
            "case_id": case_id,
            "customer_message": customer_message,
            "intent": intent,
            "evidence": evidence,
            "reply": response,
            "action": action,
            "reason": reason
        }])

        results_df = pd.concat(
            [results_df, new_row],
            ignore_index=True
        )

        # IMPORTANT: save after every example
        results_df.to_csv(
            OUTPUT_PATH,
            index=False
        )

    except Exception as e:
        print(f"Error: {e}")
        print("Stopping. Completed results have been saved.")
        break


print("\nReply evaluation data generated.")
print(f"Completed: {len(results_df)}/{len(df)}")
print(f"Saved to: {OUTPUT_PATH}")