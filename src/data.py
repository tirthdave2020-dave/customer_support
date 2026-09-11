import re
from pathlib import Path
import json


import pandas as pd


DATA_PATH = Path("twcs/twcs.csv")

def clean_text(text):
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].replace(
        -1, pd.NA
    )

    df["tweet_id"] = df["tweet_id"].astype(int)

    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"],
        errors="coerce"
    )

    return df


def build_amazon_cases(df):
    tweet_lookup = df.set_index("tweet_id").to_dict("index")

    def get_conversation(start_id):
        conversation = []
        current_id = start_id

        while pd.notna(current_id):

            tweet_id = int(current_id)

            if tweet_id not in tweet_lookup:
                return None

            tweet = tweet_lookup[tweet_id]

            conversation.append({
                "text": tweet["text"],
                "inbound": tweet["inbound"]
            })

            current_id = tweet["in_response_to_tweet_id"]

        conversation.reverse()

        return conversation

    # Find AmazonHelp conversations that end with an Amazon reply
    end_tweets = df[df["response_tweet_id"].isna()]

    amazon_end_tweets = end_tweets[
        end_tweets["author_id"] == "AmazonHelp"
    ]

    cases = []

    for conversation_id, tweet_id in enumerate(
        amazon_end_tweets["tweet_id"]
    ):

        conversation = get_conversation(tweet_id)

        if not conversation:
            continue

        customer_messages = [
            clean_text(message["text"])
            for message in conversation
            if message["inbound"]
        ]

        amazon_replies = [
            clean_text(message["text"])
            for message in conversation
            if not message["inbound"]
        ]

        if customer_messages and amazon_replies:

            cases.append({
                "conversation_id": conversation_id,
                "customer_messages": customer_messages,
                "amazon_replies": amazon_replies,
                "conversation": conversation
            })

    return cases


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data()

    print(f"Loaded {len(df):,} tweets")

    print("Building AmazonHelp conversations...")

    cases = build_amazon_cases(df)

    print(f"Built {len(cases):,} AmazonHelp cases")
    with open("amazon_cases.json", "w", encoding="utf-8") as f:
      json.dump(cases, f, ensure_ascii=False)