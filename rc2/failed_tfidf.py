import os
import pandas as pd
import numpy as np
import sys
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import src.parse_args as parse_args
import src.data_loader as data_loader
import src.model as SVD_model

# global variables
test_size = 0.2
random_state = 42


ratings_file, content_file, targets_file = sys.argv[1], sys.argv[2], sys.argv[3]

targets, train_all, new_model = True, True, True

# Parse args
# args = parse_args.parse()

# Load data
data, df_ratings = data_loader.load_ratings(ratings_file=ratings_file)
df_content = data_loader.load_content(content_file=content_file)

# Divide into data into test and train
if train_all:
    print('Using all data to train')
    test_size = 0
    train_data = data.build_full_trainset()
else:
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)


# model
model = SVD_model.RecommenderSVD()

if not new_model:
    print('Loading model from file')
    model.load_model('model/trained_model.pkl')

else:
    print('Training new model')
    model.train(train_data, n_factors=150, n_epochs=25)

    model_filename = 'model/trained_model.pkl'
    model.save_model(model_filename)
    print(f'Model saved to {model_filename}')


if not train_all:
    print('Evaluating model')
    preds, rmse_value, mae_value = model.test(test_data)

    print(f'RMSE: {rmse_value}, MAE: {mae_value}')


if targets:
    print('Generating predictions for targets')
    
    df_targets = pd.read_csv('data/targets.csv')
    
    # Convert df_targets to a list of tuples for prediction
    test_data = list(zip(df_targets['UserId'], df_targets['ItemId'], [None] * len(df_targets)))

    # Make predictions
    predictions = [model.model.predict(uid, iid, r_ui=verdict, verbose=False) for (uid, iid, verdict) in test_data]
    model_preds = np.array([pred.est for pred in predictions])

    # lookup item info
    lookup_table_item_info = df_content.set_index('ItemId')[['imdbVotes', 'Metascore', 'rtRating', 'imdbRating', 'Awards']].to_dict(orient='index')

    # Load user_repr matrices
    user_repr = np.load('matrices/user_representations_matrix.npy')
    user_mean_rating = np.load('matrices/user_mean_ratings_array.npy')

    # calculate tfidf
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_content['Plot'])

    # userId lookup table
    users = df_ratings['UserId'].unique()
    lookup_table_user = {user_id: idx for idx, user_id in enumerate(users)}

    # itemId lookup table
    items = df_content['ItemId'].unique()
    lookup_table_item = {item: idx for idx, item in enumerate(items)}

    # The final rating will be a weighted sum of some features
    final_preds = []
    for i in tqdm(range(len(model_preds))):
        itemId = predictions[i].iid
        item_info = lookup_table_item_info[itemId]
        rating = 0.25 * model_preds[i] * \
                 0.7 * item_info['imdbVotes'] * \
                 0.02 * item_info['Metascore'] * \
                 0.02 * item_info['rtRating'] * \
                 0.03 * item_info['imdbRating'] + \
                 6 * item_info['Awards']
        # np.clip(rating, 0, 10)

        line = df_targets.loc[i]
        idx_user = lookup_table_user[line['UserId']]
        idx_item = lookup_table_item[line['ItemId']]

        array_user = user_repr[idx_user]
        array_item = tfidf_matrix[idx_item]
        cos_sim = cosine_similarity(array_user, array_item)[0][0]
        rating_user_repr = cos_sim * user_mean_rating[idx_user]

        rating = rating + 0.3 * rating_user_repr

        final_preds.append(rating)

    df_targets['Rating'] = final_preds

    # normalize col rating
    min_rating = df_targets['Rating'].min()
    max_rating = df_targets['Rating'].max()
    df_targets['Rating'] = 0 + ((df_targets['Rating'] - min_rating) * (10 - 0)) / (max_rating - min_rating)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
        # Sort the DataFrame by UserId and then by Rating in descending order
    df_sorted = df_targets.sort_values(by=['UserId', 'Rating'], ascending=[True, False])

    df_sorted.to_csv('results/target_predictions_sorted.csv', index=False)

    # Drop the Rating column as it's not needed in the final output
    df_final = df_sorted.drop('Rating', axis=1)

    # Write to a CSV file
    df_final.to_csv('results/sorted_items_per_user.csv', index=False)