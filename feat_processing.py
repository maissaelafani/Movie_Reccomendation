import argparse
import numpy as np

# Loading and putting the features in clean separate columns
# pb with the data : not formatted the same way, like blank space at the end or no year data
def load_sep_feat(data):
    '''
    Input :
     data : npy

    Output :
     features_year : data with the features in different columns
     all_genres : list of all existing genres
     all_years : list of all existing years

    (for now adapted to the format of year and genres
    but if I want it clean I should make it more flexible to all kinds of features)
    '''

    features = np.load(data)

    n_movies,p = features.shape
    all_genres = []

    # Create a dictionary with all the genres
    for i in range(n_movies):
        genres = features[i][1].split('|')
        for genre in genres:
            all_genres.append(genre)
    all_genres = list(set(all_genres))

    features_year = np.empty((n_movies, p+1), dtype=object)

    # separate the year in the 1st column, create a list of the genres in the 2nd col
    for i in range(n_movies):
        title, genres = features[i]

        # cleaning : if there is a blank space at the end of the title, remove it
        if title[-1] == ' ':
            title = title[:-1]

        year = title[-6:]
        # the year is at the end, btw parentheses
        # if it's the case, proceed
        # otherwise, the year is not in the title
        if year[0] == '(' and year[-1] == ')':
            year = year[1:-1]
            features_year[i][1] = year
            title = title[:-7]
        genres = genres.split('|')

        features_year[i][0] = title
        features_year[i][2] = []

        for genre in genres:
            features_year[i][2].append(genre)


    all_years = list(set(features_year[:,1]))

    # honestly we may not need this but it's for my own sanity lol
    all_years = np.sort([i for i in all_years if i is not None])

    return features_year, all_genres, all_years


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")
    args = parser.parse_args()

    features_year, all_genres, all_years = load_sep_feat('./data/namesngenre.npy')

    table = np.load('./data/ratings_train.npy')
    table = np.nan_to_num(table)
    print(table.nonzero())

