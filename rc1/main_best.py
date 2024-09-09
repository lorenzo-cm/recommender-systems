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


def mini_batch_funkSVD_with_bias(df=df, k=20, learning_rate=0.02, regularization=0.4, epochs=12, batch_size=128):
    m, n = len(unique_users), len(unique_items)

    # funciona pior
    # P = np.random.rand(m, k)
    # Q = np.random.rand(n, k)

    P = np.random.normal(scale=1./k, size=(m, k))
    Q = np.random.normal(scale=1./k, size=(n, k))
    
    # não funciona tão bem
    # bu = np.random.randn(m)
    # bi = np.random.randn(n)

    bu = np.zeros(m)
    bi = np.zeros(n)

    # global bias
    mu = df['Rating'].mean()

    ratings = df[['User', 'Item', 'Rating']].values

    num_batches = len(ratings) // batch_size + (len(ratings) % batch_size != 0)

    for epoch in range(epochs):

        for batch_num in range(num_batches):
            start = batch_num * batch_size
            end = start + batch_size
            mini_batch = ratings[start:end]
            np.random.shuffle(mini_batch)

            for user, item, rating in mini_batch:
                i = user_to_index[user]
                j = item_to_index[item]

                error = rating - (mu + bu[i] + bi[j] + P[i, :].dot(Q[j, :].T))

                bu[i] += learning_rate * (error - regularization * bu[i])
                bi[j] += learning_rate * (error - regularization * bi[j])

                # mais rapido fazer com : do que iterando sobre k
                P[i, :] += learning_rate * (error * Q[j, :] - regularization * P[i, :])
                Q[j, :] += learning_rate * (error * P[i, :] - regularization * Q[j, :])

        learning_rate *= 0.95
        regularization *= 0.95

    return P, Q, bu, bi, mu


def predict_rating_with_bias(user_id, item_id, P, Q, bu, bi, mu, user_to_index, item_to_index):
    user_idx = user_to_index[user_id]
    item_idx = item_to_index[item_id]
    return mu + bu[user_idx] + bi[item_idx] + P[user_idx, :].dot(Q[item_idx, :].T)


def get_predictions_with_bias(targets, P, Q, bu, bi, mu, user_to_index, item_to_index):
    predictions = []

    for _, row in targets.iterrows():
        user = row['User']
        item = row['Item']

        predicted = predict_rating_with_bias(user, item, P, Q, bu, bi, mu, user_to_index, item_to_index)

        predictions.append(predicted)

    return predictions


# calculate pred matrix
a, b, bu, bi, mu = mini_batch_funkSVD_with_bias()

# get preds for the targets df
targets['Rating'] = np.clip(get_predictions_with_bias(targets, a, b, bu, bi, mu, user_to_index, item_to_index), 0, 5)

# write preds
print(targets[['UserId:ItemId', 'Rating']].to_csv(index=False))