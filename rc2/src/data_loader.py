import pandas as pd
import numpy as np

from surprise import Dataset, Reader

def load_ratings(ratings_file):
    
    df_ratings = pd.read_json(ratings_file, lines=True)

    # Define a reader with the rating scale
    reader = Reader(rating_scale=(min(df_ratings['Rating']), max(df_ratings['Rating'])))

    # Load the dataset into Surprise
    return  Dataset.load_from_df(df_ratings[['UserId', 'ItemId', 'Rating']], reader), df_ratings

def load_content(content_file):
    df_content = pd.read_json(content_file, lines=True)

    # Getting the Rotten Tomatoes ratings
    rt_ratings = []
    for ratings_list in df_content['Ratings']:
        rt_rating = next((item['Value'] for item in ratings_list if item['Source'] == 'Rotten Tomatoes'), None)
        if rt_rating:
            rt_rating = int(rt_rating[:-1])
        rt_ratings.append(rt_rating)
    df_content['rtRating'] = rt_ratings

    # Getting useful columns
    data_content = df_content[['ItemId', 'Metascore', 'imdbRating', 'imdbVotes', 'rtRating', 'Awards']].copy()

    # Updating 'Awards' column
    data_content['Awards'] = data_content['Awards'].apply(lambda x: 0 if x == 'N/A' else 1)

    # Replacing string 'N/A' with np.nan and removing number separators
    data_content = data_content.replace('N/A', np.nan)
    data_content['imdbVotes'] = data_content['imdbVotes'].str.replace(',', '')

    # Converting to numeric data
    data_content['Metascore'] = data_content['Metascore'].astype('float32')
    data_content['imdbRating'] = data_content['imdbRating'].astype('float32')
    data_content['imdbVotes'] = data_content['imdbVotes'].astype('float32')
    
    # Substitute NaN with mean
    quantiles = data_content.quantile(0.5, numeric_only=True)
    data_content = data_content.fillna(quantiles)
    
    # Normalizing imdbRating between 0 and 10
    for col in data_content.columns:
        if col in ['ItemId', 'Awards']:
            continue
        min_rating = data_content[col].min()
        max_rating = data_content[col].max()
        data_content[col] = 0 + ((data_content[col] - min_rating) * (10 - 0)) / (max_rating - min_rating)

    return data_content
