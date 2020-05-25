from data_modeling.data_preprocessing.vectorize import count_vectorize, transform_tfidf
import pandas as pd


def test_count_vectorize(article_col, X_count_df):
    count_vect, X_count = count_vectorize(article_col, ngram_range=(1, 1))
    actual_X_count_df = pd.DataFrame(X_count.toarray())
    actual_X_count_df.columns = count_vect.get_feature_names()
    assert actual_X_count_df.equals(X_count_df)


def test_transform_tfidf(X_count_df, X_tfidf_df):
    actual_X_tfidf = transform_tfidf(X_count_df)
    actual_X_tfidf_df = pd.DataFrame(actual_X_tfidf.toarray())
    assert actual_X_tfidf_df.equals(X_tfidf_df)
