from surprise import Dataset, Reader, SVD
from surprise.accuracy import rmse, mae
import os
import pickle

class RecommenderSVD:
    def __init__(self) -> None:
        self.model = None
            
    def load_model(self, file):
        if os.path.exists(file):
            with open(file, 'rb') as file:
                self.model = pickle.load(file)
                return self.model
        else:
            raise FileExistsError('File not found')

    def save_model(self, filename):
        directory = os.path.dirname(filename)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def train(self, data, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, biased=True):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, biased=biased)
        self.model.fit(data)
        return self.model
    
    def test(self, data):
        preds = self.model.test(data)
        return preds, rmse(preds, verbose=False), mae(preds, verbose=False)