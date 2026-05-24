import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # NEW: Import NRC Emotion Lexicon package
# from nrclex import NRCLex
# # ---------------------------------

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

# # NEW: Function for extracting NRC emotion scores from each comment
# def get_emotions(text):
#     emotion = NRCLex(str(text))

#     emotions = [
#         "anger", "fear", "anticipation", "trust",
#         "surprise", "sadness", "joy", "disgust"
#     ]

#     scores = {}

#     for e in emotions:
#         scores[e] = emotion.top_emotions.count(e)

#     return pd.Series(scores)
# # ------------------------------------------------

def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    comments_df = pd.read_csv("data/processed/cleaned_comments.csv")

    analyzer = SentimentIntensityAnalyzer()

    comments_df[["sentiment_score", "sentiment_label"]] = comments_df["clean_text"].apply(
        lambda text: pd.Series(classify_sentiment(text, analyzer))
    )

    #  # NEW: Apply NRC Emotion Lexicon analysis to each cleaned comment
    # emotion_df = comments_df["clean_text"].apply(get_emotions)

    # # NEW: Add emotion columns to the main comments dataframe
    # comments_df = pd.concat([comments_df, emotion_df], axis=1)
    # #------------------------------------------------------
    
     
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

#  # NEW: Reshape NRC emotion data so it can be plotted
#     emotion_melted = emotion_summary.melt(
#         id_vars="brand_category",
#         var_name="emotion",
#         value_name="count"
#     )

    # # NEW: Create NRC emotion comparison graph
    # plt.figure(figsize=(11, 6))
    # sns.barplot(
    #     data=emotion_melted,
    #     x="emotion",
    #     y="count",
    #     hue="brand_category"
    # )
    # plt.title("NRC Emotion Lexicon Comparison by Brand Category")
    # plt.xlabel("Emotion")
    # plt.ylabel("Emotion Frequency")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig("outputs/figures/emotion_comparison.png", dpi=300)
    # plt.close()

    # print("Saved sentiment and emotion analysis results.")
    # print(sentiment_summary)
    # print(emotion_summary)
    # # ---------------------------------------------------------

    print("Saved sentiment results.")
    print(sentiment_summary)
    


if __name__ == "__main__":
    main()