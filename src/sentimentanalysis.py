import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def classify_sentiment(text, analyzer):
    """
    Uses VADER compound score to classify comment sentiment.
    """

    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return score, label


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    comments_df = pd.read_csv("data/processed/cleaned_comments.csv")

    analyzer = SentimentIntensityAnalyzer()

    comments_df[["sentiment_score", "sentiment_label"]] = comments_df["clean_text"].apply(
        lambda text: pd.Series(classify_sentiment(text, analyzer))
    )

    comments_df.to_csv("data/processed/comments_with_sentiment.csv", index=False)

    sentiment_summary = (
        comments_df
        .groupby(["brand_category", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    sentiment_summary.to_csv("outputs/tables/sentiment_summary.csv", index=False)

    sentiment_percent = (
    comments_df
    .groupby(["brand_category", "sentiment_label"])
    .size()
    .reset_index(name="count"))
    sentiment_percent["percentage"] = sentiment_percent.groupby("brand_category")["count"].transform(
    lambda x: round((x / x.sum()) * 100, 2))
    sentiment_percent.to_csv("outputs/tables/sentiment_percentage_summary.csv", index=False)
    print(sentiment_percent)

    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=comments_df,
        x="brand_category",
        hue="sentiment_label",
        order=["Samsung", "iPhone", "Comparison", "Other"]
    )
    plt.title("Sentiment Comparison: Samsung vs iPhone YouTube Comments")
    plt.xlabel("Brand Category")
    plt.ylabel("Number of Comments")
    plt.tight_layout()
    plt.savefig("outputs/figures/sentiment_comparison.png", dpi=300)
    plt.close()

    print("Saved sentiment results.")
    print(sentiment_summary)
    


if __name__ == "__main__":
    main()