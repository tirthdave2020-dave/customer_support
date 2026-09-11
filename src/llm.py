import os
from dotenv import load_dotenv
from google import genai

from src.retrivel import retrieve_cases


INTENTS = [
    "delivery_issue",
    "return_replacement",
    "refund_issue",
    "payment_issue",
    "prime_membership",
    "product_issue",
    "product_information",
    "technical_issue",
    "account_issue",
    "seller_marketplace",
    "security_fraud",
    "feedback_complaint"
]


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def classify_intent(customer_message):

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"""
You are an intent classifier for an Amazon customer support system.

Classify the customer's message into exactly ONE of these intents:

{INTENTS}

Return ONLY the intent name.
Do not explain your answer.
Do not add punctuation.

Customer message:
{customer_message}
"""
    )

    return response.text.strip()


def get_relevant_cases(customer_message, k=5):

    return retrieve_cases(
        customer_message,
        k=k
    )


def generate_response(customer_message, intent, results):

    # Security issues should always be escalated.
    if intent == "security_fraud":

        return {
            "reply": (
                "Please contact Amazon Support through your account "
                "so the unauthorized activity can be investigated securely."
            ),
            "action": "escalate",
            "reason": (
                "Security or unauthorized activity requires "
                "human investigation."
            )
        }


    # Potentially unauthorized payment
    if intent == "payment_issue" and any(
        word in customer_message.lower()
        for word in [
            "unauthorized",
            "fraud",
            "scam",
            "didn't make",
            "did not make",
            "don't recognize",
            "do not recognize"
        ]
    ):

        return {
            "reply": (
                "Please contact Amazon Support through your account "
                "so the payment issue can be investigated securely."
            ),
            "action": "escalate",
            "reason": (
                "A potentially unauthorized or disputed payment "
                "requires human investigation."
            )
        }


    # Build historical evidence
    evidence = ""

    for i, result in enumerate(results, 1):

        case = result["case"]

        evidence += f"""
Historical Case {i}:
Customer:
{" ".join(case["customer_messages"])}

AmazonHelp:
{" ".join(case["amazon_replies"])}

Similarity:
{result["score"]:.4f}

"""


    # Generate grounded response
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"""
You are an Amazon customer support agent.

Draft a helpful response to the customer using the
historical AmazonHelp cases provided as evidence.

Rules:
- Use the historical responses as guidance.
- Do not invent policies, refunds, guarantees, or actions.
- Do not claim an action has been completed.
- Be concise and professional.
- Respond in the same language as the customer when possible.

Escalation rules:
- Escalate if the evidence is not sufficiently relevant.
- Escalate unusual or ambiguous requests.
- Escalate issues requiring case-specific investigation.
- Auto-handle common issues when historical evidence provides
  a clear response pattern.

Return exactly:

Reply: <customer-facing response>
Action: auto_handle OR escalate
Reason: <short explanation>

Customer message:
{customer_message}

Predicted intent:
{intent}

Historical evidence:
{evidence}
"""
    )

    text = response.text.strip()

    # Parse the generated output
    reply = ""
    action = "escalate"
    reason = ""

    for line in text.splitlines():

        line = line.strip()

        if line.lower().startswith("reply:"):
            reply = line.split(":", 1)[1].strip()

        elif line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip().lower()

        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()


    return {
        "reply": reply,
        "action": action,
        "reason": reason
    }