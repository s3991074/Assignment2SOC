import os
import re
import pandas as pd


def clean_text(text):
    """
    Basic text cleaning for YouTube comments.
    """

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def classify_brand(text):
    """
    Classifies each comment into Samsung, iPhone, Comparison, or Other.
    """

    text = str(text).lower()

    samsung_terms = [
        "samsung", "galaxy", "android", "s24", "s25",
        "ultra", "z fold", "z flip"
    ]

    iphone_terms = [
        "iphone", "apple", "ios", "15 pro", "16 pro",
        "pro max"
    ]

    has_samsung = any(term in text for term in samsung_terms)
    has_iphone = any(term in text for term in iphone_terms)

    if has_samsung and has_iphone:
        return "Comparison"
    elif has_samsung:
        return "Samsung"
    elif has_iphone:
        return "iPhone"
    else:
        return "Other"


def main():
    os.makedirs("data/processed", exist_ok=True)

    comments_df = pd.read_csv("data/raw/youtube_comments.csv")

    comments_df = comments_df.drop_duplicates(subset=["comment_id"])
    comments_df = comments_df.dropna(subset=["comment_text"])

    comments_df["clean_text"] = comments_df["comment_text"].apply(clean_text)
    comments_df = comments_df[comments_df["clean_text"].str.len() > 5]

    comments_df["brand_category"] = comments_df["clean_text"].apply(classify_brand)

    comments_df.to_csv("data/processed/cleaned_comments.csv", index=False)

    print("Saved cleaned comments to data/processed/cleaned_comments.csv")
    print("\nBrand category counts:")
    print(comments_df["brand_category"].value_counts())


if __name__ == "__main__":
    main()