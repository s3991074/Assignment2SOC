import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def get_top_keywords(df, brand, top_n=20):
    """
    Finds top TF-IDF keywords for a brand category.
    """

    brand_df = df[df["brand_category"] == brand]

    if len(brand_df) == 0:
        return pd.DataFrame(columns=["brand", "word", "score"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(brand_df["clean_text"])
    scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    keyword_df = pd.DataFrame({
        "brand": brand,
        "word": words,
        "score": scores
    }).sort_values(by="score", ascending=False)

    return keyword_df.head(top_n)


def run_lda_topic_model(df, number_of_topics=5):
    """
    Runs LDA topic modelling on Samsung, iPhone, and Comparison comments.
    """

    topic_df = df[df["brand_category"].isin(["Samsung", "iPhone", "Comparison"])]

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=5
    )

    doc_term_matrix = vectorizer.fit_transform(topic_df["clean_text"])

    lda = LatentDirichletAllocation(
        n_components=number_of_topics,
        random_state=42
    )

    lda.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()
    topics = []

    for topic_index, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:]]
        topics.append({
            "topic_number": topic_index + 1,
            "top_words": ", ".join(top_words)
        })

    return pd.DataFrame(topics)


def main():
    os.makedirs("outputs/tables", exist_ok=True)

    comments_df = pd.read_csv("data/processed/comments_with_sentiment.csv")

    keyword_results = []

    for brand in ["Samsung", "iPhone", "Comparison"]:
        keyword_results.append(get_top_keywords(comments_df, brand))

    keywords_df = pd.concat(keyword_results)
    keywords_df.to_csv("outputs/tables/top_keywords.csv", index=False)

    topics_df = run_lda_topic_model(comments_df, number_of_topics=5)
    topics_df.to_csv("outputs/tables/topic_model_results.csv", index=False)

    print("Saved keyword and topic results.")
    print(topics_df)


if __name__ == "__main__":
    main()