from src.llm import classify_intent, get_relevant_cases, generate_response


def run_agent(customer_message):

    intent = classify_intent(customer_message)

    # Security cases don't need retrieval
    if intent == "security_fraud":
        result = generate_response(
            customer_message,
            intent,
            []
        )

    else:
        results = get_relevant_cases(
            customer_message,
            k=5
        )

        result = generate_response(
            customer_message,
            intent,
            results
        )

    return {
        "intent": intent,
        "reply": result["reply"],
        "action": result["action"],
        "reason": result["reason"]
    }


if __name__ == "__main__":

    print("Amazon Customer Support AI Agent")
    print("--------------------------------")

    customer_message = input("\nCustomer message: ")

    result = run_agent(customer_message)

    print("\nIntent:", result["intent"])
    print("Reply:", result["reply"])
    print("Action:", result["action"])
    print("Reason:", result["reason"])