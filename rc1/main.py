import pandas as pd
import numpy as np
import sys


np.random.seed(42)

df = pd.read_csv(sys.argv[1])
targets = pd.read_csv(sys.argv[2])
df[['User', 'Item']] = df['UserId:ItemId'].str.split(':', expand=True)
targets[['User', 'Item']] = targets['UserId:ItemId'].str.split(':', expand=True)

unique_users = df['User'].unique()
unique_items = df['Item'].unique()

user_to_index = {user: idx for idx, user in enumerate(unique_users)}
item_to_index = {item: idx for idx, item in enumerate(unique_items)}


def funkSVD(df=df, k=20, learning_rate=0.08, regularization=0.1, epochs=14):
    m, n = len(unique_users), len(unique_items)
    P = np.random.rand(m, k)
    Q = np.random.rand(n, k)

    ratings = df[['User', 'Item', 'Rating']].values

    for epoch in range(epochs):
        np.random.shuffle(ratings)

        mse_accumulated = 0
        count = 0

        for user, item, rating in ratings:
            i = user_to_index[user]
            j = item_to_index[item]
            
            error = rating - P[i, :].dot(Q[j, :].T)
            P_new = learning_rate * (error * Q[j, :] - regularization * P[i, :])
            Q[j, :] += learning_rate * (error * P[i, :] - regularization * Q[j, :])
            P[i, :] += P_new


            predicted_rating = P[i, :].dot(Q[j, :].T)
            mse_accumulated += (rating - predicted_rating)**2
            count += 1
        
        learning_rate = learning_rate * 0.95
        regularization = regularization * 0.9

        mse = mse_accumulated / count
        print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.4f}")

    return P, Q


def predict_rating(user_id, item_id, P, Q, user_to_index, item_to_index):
    user_idx = user_to_index[user_id]
    item_idx = item_to_index[item_id]
    return P[user_idx, :].dot(Q[item_idx, :].T)

def get_predictions(targets, P, Q, user_to_index, item_to_index):
    predictions = []

    for _, row in targets.iterrows():
        user = row['User']
        item = row['Item']

        predicted = predict_rating(user, item, P, Q, user_to_index, item_to_index)

        predictions.append(predicted)

    return predictions

# calculate pred matrix
a, b = funkSVD()

# get preds for the targets df
targets['Rating'] = np.clip(get_predictions(targets, a, b, user_to_index, item_to_index), 0, 5)

# write preds
print(targets[['UserId:ItemId', 'Rating']].to_csv(index=False))
