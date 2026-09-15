# AI Customer Support Agent

An AI-powered customer support agent built for **AmazonHelp** using the **Customer Support on Twitter** dataset.

The system combines intent classification, semantic retrieval of historical support cases, LLM-based response generation, and escalation decisions.

---

## What the Agent Does

The agent performs three main tasks:

1. **Intent classification**  
   Classifies an incoming customer message into one of 12 support intents.

2. **Evidence-grounded response generation**  
   Retrieves similar historical AmazonHelp support cases and uses their responses as evidence when drafting a reply.

3. **Escalation decision**  
   Decides whether the request should be `auto_handle` or `escalate`, with a short reason.

The final output has the following structure:

```text
Intent: <predicted intent>
Reply: <customer-facing response>
Action: auto_handle OR escalate
Reason: <reason for the decision>
```

---

## Problem Framing

The goal is not to replace a human customer-support team.

The goal is to assist with common Amazon customer-support requests by combining historical support behavior with an LLM.

For this project, a good response should:

- Correctly identify the customer's support intent.
- Be grounded in how AmazonHelp historically responded to similar problems.
- Avoid unsupported policies, guarantees, or claims.
- Provide a useful response for common issues.
- Escalate sensitive, ambiguous, unusual, or case-specific requests.

The system therefore focuses on:

- Intent classification
- Evidence-grounded response drafting
- Escalation recommendation

### What I Chose Not to Build

I did not build:

- An autonomous system that performs refunds, replacements, cancellations, or account changes.
- A system that claims an action has been completed when it has not.
- A fine-tuned customer-support LLM.
- A complete production customer-support platform with authentication, monitoring, human-agent tooling, and persistent customer state.

The result is a **decision-support and response-drafting prototype**, rather than a production customer-service system.

---

## Architecture

```text
                    Customer Message
                           |
                           v
                  Intent Classification
                           |
                           v
                  Historical Retrieval
                           |
                           v
                 Gemini Response Generation
                           |
                           v
              +------------+------------+
              |            |            |
            Intent       Reply        Action
                                      |
                              auto_handle / escalate
                                      |
                                    Reason
```

---

## Dataset

The project uses the **Customer Support on Twitter** dataset from Kaggle:

- Approximately 2.8 million tweets
- Multiple brands
- Multi-turn customer-support interactions
- Informal, noisy, and multilingual text

Only **AmazonHelp** interactions are used in this project.

Historical conversations are reconstructed using the tweet response relationships in the dataset, particularly `in_response_to_tweet_id`.

After cleaning and filtering, the project constructed:

**83,888 AmazonHelp support cases**

A support case contains the customer's messages and the corresponding AmazonHelp responses.

---

## Intent Taxonomy

The system uses a fixed taxonomy of 12 intents:

1. `delivery_issue`
2. `return_replacement`
3. `refund_issue`
4. `payment_issue`
5. `prime_membership`
6. `product_issue`
7. `product_information`
8. `technical_issue`
9. `account_issue`
10. `seller_marketplace`
11. `security_fraud`
12. `feedback_complaint`

A fixed taxonomy was chosen instead of allowing the LLM to generate arbitrary intent names so that intent predictions can be evaluated consistently.

---

## Retrieval

Historical support cases are represented as:

```text
Customer problem
+
AmazonHelp response
```

Each complete support case is treated as one retrieval unit rather than indexing individual tweets independently.

The cases are embedded using:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

FAISS with inner-product similarity is used for nearest-neighbor retrieval.

The embeddings are normalized, making inner product equivalent to cosine similarity.

For each incoming message, the system retrieves the **top 5 similar historical cases**.

These cases are then provided to Gemini as evidence for response generation.

### Demo Retrieval Corpus

The full development corpus contains 83,888 cases.

For the runnable repository, a smaller **5,000-case retrieval artifact** is included so that the project can be run without downloading or regenerating the full embedding matrix.

The demo artifacts are:

```text
artifacts/
├── demo_cases.json
└── demo_embeddings.npy
```

The original full embedding file is intentionally not included in the repository because of its size.

---

## Response Generation

Gemini is used to generate the customer-facing response.

The model receives:

- Customer message
- Predicted intent
- Top 5 retrieved historical cases

The response-generation prompt instructs the model to:

- Use historical cases as evidence.
- Avoid inventing policies or guarantees.
- Avoid claiming that an action has been completed.
- Produce concise and professional responses.
- Respond in the customer's language when possible.
- Escalate ambiguous or sensitive requests.
- Auto-handle common issues when historical evidence provides a clear response pattern.

### Explicit Safety Handling

Security and potentially unauthorized payment cases are explicitly escalated.

For example, messages involving:

- Unauthorized purchases
- Fraud
- Scams
- Suspicious activity
- Account security concerns

are not treated as ordinary auto-handle cases.

---

## Installation

Clone the repository and enter the project directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## API Key Setup

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Each user should provide their own Gemini API key.

The repository does not contain an API key.

**Do not commit `.env` or your API key to GitHub.**

---

## Running the Agent

From the project root:

```bash
python -m streamlit run app.py

Enter a customer message when prompted.

Example:

```text
My package has not arrived yet
```

The agent returns:

```text
Intent: delivery_issue
Reply: ...
Action: auto_handle
Reason: ...
```

---

# Evaluation

## Evaluation Methodology

The evaluation focuses on the three main components:

1. Intent classification
2. Escalation decision
3. Reply quality

Because the intent classes are imbalanced, **macro F1** is reported alongside accuracy.

---

## Intent Classification

A 200-example golden evaluation set was created from the reconstructed AmazonHelp cases.

The examples were labelled using the fixed 12-intent taxonomy.

The labeling rule was:

> If a message contains multiple issues, label it according to the latest or currently unresolved customer request.

The initial sampled labels contained inconsistencies and noisy assignments. The evaluation set was cleaned using the defined intent taxonomy.

The resulting labels should be considered **AI-assisted evaluation labels rather than a fully independent multi-annotator human benchmark**.

### Results

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 28.1% | 3.7% |
| TF-IDF + Logistic Regression | 27.5% | 3.6% |
| Gemini intent classifier | **64.0%** | **53.7%** |

The majority-class baseline always predicts the most frequent intent.

The TF-IDF baseline represents customer messages using TF-IDF features and predicts the intent using Logistic Regression.

The Gemini classifier substantially outperformed both simple baselines on the evaluation set.

**Important:** The 64.0% accuracy and 53.7% macro F1 measure **intent classification only**. They are not overall agent accuracy or reply-quality scores.

---

## Escalation Evaluation

A small interim evaluation set was created to test the `auto_handle` versus `escalate` decision.

Seven examples were completed before the Gemini free-tier request quota was reached.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 71.4% |
| Macro F1 | 70.8% |

Because the sample contains only seven completed examples, these numbers are treated as an **engineering sanity check rather than a statistically reliable benchmark**.

---

## Reply Quality Evaluation

Two generated responses were manually reviewed using five criteria:

1. Correctness
2. Evidence grounding
3. Helpfulness
4. Professionalism
5. Unsupported claims

The two responses received an approximate average qualitative score of:

**4.0 / 5**

This is only a qualitative sanity check because the sample is extremely small.

A larger evaluation should use an LLM-as-judge rubric calibrated against multiple independent human reviewers.

---

## Baseline Comparison

The intent-classification results show that the final Gemini classifier performs substantially better than the two simple reference baselines.

```text
Majority baseline       28.1%
TF-IDF + Logistic       27.5%
Gemini                  64.0%
```

The comparison suggests that the LLM-based classifier captures semantic intent information that the simple lexical baseline does not capture effectively on this noisy support dataset.

However, the benchmark has important limitations described below.

---

# Failure Analysis

## 1. Similar intents are difficult to distinguish

Some messages can reasonably belong to multiple related categories, such as:

- Delivery vs. product issue
- Return vs. refund
- Payment vs. security/fraud

### Example

A customer may mention both a missing package and a refund request in the same message.

### Hypothesis

The taxonomy boundaries can be ambiguous, especially for short informal messages.

### Improvement

Define clearer decision boundaries and consider hierarchical classification for closely related intents.

---

## 2. Noisy customer language

The dataset contains:

- Spelling mistakes
- Abbreviations
- Informal language
- Emojis
- Mentions
- Incomplete sentences

Example:

```text
my parcel is not deliverd from last two days delivery date was gone
```

The intended meaning is clear to a human despite the spelling and grammatical errors.

### Hypothesis

Noisy language can reduce both classification and retrieval quality.

### Improvement

Add stronger normalization and explicitly evaluate robustness on noisy customer messages.

---

## 3. Multilingual interactions

The dataset contains messages in multiple languages.

The multilingual embedding model helps semantic retrieval, but language-specific intent boundaries and response generation can still be challenging.

### Hypothesis

The embedding model improves cross-language retrieval, but the LLM may occasionally produce an inappropriate response language or style.

### Improvement

Add language detection and enforce the customer's language during response generation.

---

## 4. Partially relevant retrieval results

Semantic search can retrieve a case that is related to the customer's general topic but does not exactly match the issue.

For example, a delivery query may retrieve another delivery-related case involving a damaged package.

### Hypothesis

Embedding similarity captures broad semantic similarity but does not always capture operational details.

### Improvement

Combine semantic retrieval with intent filtering, metadata filtering, and minimum similarity thresholds.

---

## 5. Evaluation and retrieval data overlap

The evaluation examples originate from the same historical dataset used to construct the retrieval corpus.

Therefore, the retrieval system may encounter the exact evaluation case or a highly similar historical case.

### Hypothesis

This can make response generation appear better than it would be on genuinely unseen customer messages.

### Improvement

Create a strict train/retrieval versus evaluation split before constructing the retrieval index.

---

# What Is Misleading About My Headline Number?

The most potentially misleading number is the:

**64.0% intent-classification accuracy**

Although it is substantially higher than the two simple baselines, it should **not** be interpreted as a production accuracy estimate.

There are several reasons:

- The evaluation set contains only 200 examples.
- The classes are imbalanced.
- The labels were AI-assisted rather than produced through independent multi-annotator labelling.
- The evaluation examples originate from the same historical dataset used to construct the retrieval corpus.
- Exact or near-duplicate historical cases may therefore be available during retrieval.

The number is best interpreted as an **initial prototype benchmark**, not as a claim that the system will correctly classify 64% of unseen production messages.

A stronger experiment would create a strict held-out evaluation set that is excluded from the retrieval corpus before embeddings are generated.

---

# Limitations

The current prototype has several limitations:

- Intent evaluation contains only 200 examples.
- Intent classes are imbalanced.
- The evaluation labels were AI-assisted.
- Reply-quality evaluation contains only two manually reviewed responses.
- Escalation evaluation contains only seven completed examples.
- The retrieval corpus and evaluation examples are not strictly separated.
- Only AmazonHelp was evaluated.
- The system does not execute customer-service actions.
- API rate limits prevented a larger response-generation evaluation.
- The current evaluation does not provide statistically strong evidence for production deployment.

---

# What I Would Do With One More Week

With another week, I would prioritize **evaluation quality and reliability** rather than adding more features.

## 1. Build a strictly held-out benchmark

Split the historical cases before creating embeddings so that evaluation cases can never be retrieved as evidence.

## 2. Improve human annotation

Create approximately 250 carefully sampled examples and have multiple human reviewers independently label them.

Measure inter-annotator agreement and resolve disagreements using predefined rules.

## 3. Evaluate retrieval separately

Measure:

- Recall@1
- Recall@5
- Similarity score distributions
- Retrieval performance by intent

## 4. Improve escalation

Create a larger balanced escalation dataset and evaluate:

- Precision
- Recall
- F1
- False auto-handle rate
- False escalation rate

For a support system, false auto-handling of sensitive cases is especially important to monitor.

## 5. Build an LLM-as-judge evaluation

Create a structured rubric covering:

- Correctness
- Evidence grounding
- Helpfulness
- Professionalism
- Unsupported claims

Then compare LLM-judge ratings against independent human ratings and report agreement.

## 6. Improve retrieval confidence

Introduce similarity thresholds and intent-aware retrieval so that weakly related historical cases do not automatically become evidence.

## 7. Test realistic production-like messages

Evaluate:

- Spelling errors
- Multilingual messages
- Very short messages
- Multiple simultaneous issues
- Unseen problems
- Ambiguous requests
- Security-sensitive requests

---

# Project Structure

```text
customer_support/
│
├── app.py
├── README.md
├── decision_log.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data.py
│   ├── intent.py
│   ├── retrivel.py
│   └── llm.py
│
├── artifacts/
│   ├── demo_cases.json
│   └── demo_embeddings.npy
│
├── eval/
│   ├── golden_set_proposed_cleaned (2).csv
│   ├── escalation_golden_set_30.csv
│   ├── escalation_eval_results.csv
│   ├── reply_eval_results.csv
│   ├── intent_eval.py
│   ├── esca.py
│   └── reply.py
│
└── twcs/
    └── twcs.csv
```

The full raw dataset and full development embedding matrix are not required for the lightweight demo repository.

---

# Decision Log

The major non-obvious engineering decisions are documented separately in:

```text
decision_log.md
```

Key decisions include:

- Selecting AmazonHelp as the target brand.
- Reconstructing support cases from tweet relationships.
- Using complete support cases as retrieval units.
- Including both customer problems and historical AmazonHelp responses in retrieval documents.
- Using a fixed 12-intent taxonomy.
- Selecting a multilingual embedding model.
- Using FAISS with normalized embeddings.
- Retrieving the top five historical cases.
- Using RAG rather than fine-tuning an LLM.
- Explicitly escalating security and potentially unauthorized payment cases.
- Comparing against simple baselines.
- Reporting evaluation limitations and possible retrieval/evaluation overlap.

---

# Reproducibility Notes

The repository includes a 5,000-case retrieval artifact so that the agent can run without regenerating the full 83,888-case embedding corpus.

To reproduce the lightweight demo:

```bash
pip install -r requirements.txt
```

Create:

```text
.env
```

with:

```text
GEMINI_API_KEY=your_api_key_here
```

Then:

```bash
python -m streamlit run app.py

The Gemini API key is user-provided and is not included in this repository.