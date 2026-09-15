from src.llm import (
    classify_intent,
    get_relevant_cases,
    generate_response
)

print("Amazon Customer Support AI Agent")
print("----------------------------------")

customer_message = input("Enter customer message: ")

print("\nClassifying intent...")
intent = classify_intent(customer_message)

print(f"Intent: {intent}")

print("\nRetrieving relevant historical cases...")

if intent == "security_fraud":
    results = []
else:
    results = get_relevant_cases(
        customer_message,
        k=5
    )

print(f"Retrieved {len(results)} historical cases.")

print("\nGenerating response...")

result = generate_response(
    customer_message,
    intent,
    results
)

print("\n----------------------------------")
print("FINAL RESULT")
print("----------------------------------")

print(f"\nIntent: {intent}")

print(f"\nReply:\n{result['reply']}")

print(f"\nAction: {result['action']}")

print(f"\nReason:\n{result['reason']}")

if results:
    print("\n----------------------------------")
    print("HISTORICAL EVIDENCE")
    print("----------------------------------")

    for i, item in enumerate(results, 1):
        case = item["case"]

        print(f"\nCase {i}")
        print(f"Similarity: {item['score']:.4f}")

        print("\nCustomer:")
        print(" ".join(case["customer_messages"]))

        print("\nAmazonHelp:")
        print(" ".join(case["amazon_replies"]))