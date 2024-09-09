import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def generate_user_item_predictions(df_c):
    """
    Generates user-item rating predictions based on content similarity.

    Args:
        ratings_file (str): Path to the JSONL file containing user ratings.
        content_file (str): Path to the JSONL file containing item content.

    Returns:
        DataFrame: DataFrame with user-item rating predictions.
    """
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_c['Plot'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return tfidf_df



def calc_similarity(tfidf_df, df_c, itemId1, itemId2):
    line1 = df_c[df_c['ItemId'] == itemId1].index[0]
    line2 = df_c[df_c['ItemId'] == itemId2].index[0]
    return cosine_similarity(tfidf_df.loc[line1].values.reshape(1, -1), tfidf_df.loc[line2].values.reshape(1, -1))



def calc_sim_user_item(tfidf_df, df_c, df_r, userId, itemId):
    aux = df_r[df_r['UserId'] == userId]
    sims = []
    for item_id, rating in zip(aux['ItemId'], aux['Rating']):
        sims.append(calc_similarity(tfidf_df, df_c, item_id, itemId)[0][0] * rating)
    return sum(sims)
        


if __name__ == '__main__':
    df_r = pd.read_json('../data/ratings.jsonl', lines=True).drop('Timestamp', axis=1)
    df_c = pd.read_json('../data/content.jsonl', lines=True)
    i_df = generate_user_item_predictions(df_c)


    preds = []
    df_t = pd.read_csv('../data/targets.csv')
    for i in tqdm(range(len(df_t))):
        preds.append(calc_sim_user_item(i_df, df_c, df_r, df_t["UserId"][i], df_t["ItemId"][i]))


    df_t['Rating'] = preds
    df_t.to_csv('results/test.csv', index=False)
    
