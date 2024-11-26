import argparse
import numpy as np
from feat_processing import load_sep_feat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")
    args = parser.parse_args()

# Open Ratings table
table = np.load(args.name) ## DO NOT CHANGE THIS LINE

# Data preprocessing
table = np.nan_to_num(table)

# Load features

add_features = load_sep_feat('./data/namesngenre.npy')

# TODO : think of an optimized way to learn the features in the latent space then map them
# en vrai instead of learning for each one, learn for the few existing distinct ones
# then put them together once the training is done
# I think that's not how the paper does it but it's sth to consider ?

# Model definition
class Model:
    def __init__(self, parameters, feat = True, init_feat = 'random', k = 0):
        self.parameters = parameters
        self.feat = feat

        if k == 0:
            k = self.parameters['features']

        self.nb_users, self.nb_movies = self.parameters['Dataset'].shape

        # this will be filled if feat is True
        self.F = np.zeros((self.nb_movies,k))
        
        if self.feat:
            self.table_feat = self.parameters['add_features'][0]
            self.all_feat = self.parameters['add_features'][1:] # here it's just all_years and all_genres
            self.nbs_feat = [len(i) for i in self.all_feat] # |Ft| the number of possibility for each feature

            # init to random or init to 0?
            # Y the latent factor representation of the features
            self.Y = []
            for t in range(len(self.all_feat)):
                # for each feature, we create a matrix of size |Ft| x k
                if init_feat == 'zero' :
                    self.Y.append(np.zeros((self.nbs_feat[t], k)))
                else: # random by default
                    self.Y.append(np.random.normal(scale=1. / k, size=(self.nbs_feat[t], k)))

        # Take all the non zero values of the dataset and separate the row and column indices
        self.sample_row, self.sample_col = self.parameters['Dataset'].nonzero()
        self.U_bias, self.I_bias = np.zeros(self.nb_users), np.zeros(self.nb_movies)

        self.global_bias = np.mean(self.parameters['Dataset'][np.where(self.parameters['Dataset'] != 0)])

        self.U, self.I = (np.random.normal(scale=1. / self.parameters['features'], size=(self.nb_users, self.parameters['features'])),
                          np.random.normal(scale=1. / self.parameters['features'], size=(self.nb_movies, self.parameters['features'])))
        
    def train(self):
        c = 0
        while c < self.parameters['epoch']:
            training_indices = np.arange(len(self.sample_row))
            np.random.shuffle(training_indices)

            for u, i in zip(self.sample_row[training_indices], self.sample_col[training_indices]):

                if self.feat:
                    # Compute the 'item' term including the features                
                    for t in range(len(self.all_feat)):
                        for a in range(self.nbs_feat[t]):
                            self.F[i] += self.Y[t][a] 
                        self.F[i] = self.F[i] / self.nbs_feat[t]

                pred = self.U[u].dot(self.I[i]) + self.global_bias + self.U_bias[u] + self.I_bias[i]
                e = self.parameters['Dataset'][u, i] - pred

                self.U_bias[u] += self.parameters['learning_rate'] * (e - self.parameters['regulation'] * self.U_bias[u])
                self.I_bias[i] += self.parameters['learning_rate'] * (e - self.parameters['regulation'] * self.I_bias[i])

                self.U[u] += self.parameters['learning_rate'] * (e * (self.I[i] + self.F[i]) - self.parameters['regulation']* self.U[u])
                self.I[i] += self.parameters['learning_rate'] * (e * self.U[u] - self.parameters['regulation'] * self.I[i])

                if self.feat:
                    for t in range(len(self.all_feat)):
                        for a in range(self.nbs_feat[t]):
                            self.Y[t][a] += self.parameters['learning_rate'] * (e * self.U[u] / self.nbs_feat[t] - self.parameters['regulation'] * self.Y[t][a])

            c += 1

        biases = self.global_bias + self.U_bias[:, np.newaxis] + self.I_bias[np.newaxis, :]
        predictions = self.U.dot((self.I + self.F).T) + biases

        return np.nan_to_num(predictions)


# Hyper Parameters Definition
Parameters = {'Dataset': table,
              'features': 60,
              'add_features': add_features,
              'regulation': 0.04,
              'epoch': 100,
              'learning_rate': 0.0035}

# Model Training
MyModel = Model(Parameters)
result = MyModel.train()

# Post Process
result = np.clip(result, 0.5, 5)

np.save("output.npy", result)
