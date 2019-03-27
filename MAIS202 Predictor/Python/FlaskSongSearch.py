from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pandas as pd
from flask import Flask, jsonify, request
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Load the data and split into 3 equal sets
data = pd.read_csv('SpotifyAudioFeaturesNov2018.csv', header=None)
data = data.iloc[1:-1, 3:]
valid, train = np.split(data, [int(.25*len(data))])
print(train.shape)
valid = np.array(valid)
train = np.array(train)
train_x = np.array(train[:, 0:-1]).astype(np.float)
train_y = np.array(train[:, -1]).astype(np.float)
valid_x = np.array(valid[:, 0:-1]).astype(np.float)
valid_y = np.array(valid[:, -1]).astype(np.float)

### YOUR CODE HERE - Fit a 10-degree polynomial using Sci-Kit Learn
polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(train_x)
model = LinearRegression()
model.fit(x_poly, train_y)


# n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
# for estimator in n_estimators:
#    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
#    rf.fit(train_x, train_y)
#    ### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
#    print(estimator, "\n")
#    print('rf train mse: ', mean_squared_error(rf.predict(train_x), train_y))
#    print('rf valid mse: ', mean_squared_error(rf.predict(valid_x), valid_y))

# max_depths = np.linspace(30, 100, 71, endpoint=True)
# for max_depth in max_depths:
#    rf = RandomForestClassifier(n_estimators=16,max_depth=max_depth, n_jobs=-1)
#    rf.fit(train_x, train_y)
#    ### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
#    print(max_depth, "\n")
#    print('rf train mse: ', mean_squared_error(rf.predict(train_x), train_y))
#    print('rf valid mse: ', mean_squared_error(rf.predict(valid_x), valid_y))

# min_samples_splits = np.arange(2,10)
# for split in min_samples_splits:
#    rf = RandomForestClassifier(n_estimators=16,max_depth=30, min_samples_split = split)
#    rf.fit(train_x, train_y)
#    ### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
#    print(split, "\n")
#    print('rf train mse: ', mean_squared_error(rf.predict(train_x), train_y))
#    print('rf valid mse: ', mean_squared_error(rf.predict(valid_x), valid_y))

# min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
# for leaf in min_samples_leafs:
#    rf = RandomForestClassifier(n_estimators=16,max_depth=30, min_samples_split = 2, min_samples_leaf=leaf )
#    rf.fit(train_x, train_y)
#    ### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
#    print(leaf, "\n")
#    print('rf train mse: ', mean_squared_error(rf.predict(train_x), train_y))
#    print('rf valid mse: ', mean_squared_error(rf.predict(valid_x), valid_y))

### YOUR CODE HERE - Use model to predict output of validation set
print(valid_x.shape)
y_poly_pred = model.predict(polynomial_features.fit_transform(valid_x))

### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
print('poly train mse: ', mean_squared_error(model.predict(x_poly), train_y))
print('poly valid mse: ', mean_squared_error(y_poly_pred, valid_y))



app = Flask(__name__)

client_credentials_manager = SpotifyClientCredentials(client_id='e6c6ff809f24475cbcf4280718a78889', client_secret='238f8d6a6f7d4ea3a6b2d09cb1cd65a0')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


Songs = []

@app.route('/GetSong/', methods=['GET'])
def get_song():
    title = request.args.get("title")
    results = sp.search(q='track:' + title, type='track')
    return jsonify(results)

@app.route('/SongAnalysis/', methods=['GET'])
def get_song_details():
    id = request.args.get("id")
    print(id)
    results = sp.audio_features([id])[0]
    song_features = [results["acousticness"],results["danceability"], results["duration_ms"], results["energy"],
                     results["instrumentalness"],results["key"],results["liveness"],results["loudness"],
                     results["mode"], results["speechiness"], results["tempo"], results["time_signature"],
                     results["valence"]]
    song_features = np.array(song_features)
    song_features_poly = polynomial_features.fit_transform(song_features.reshape(1,-1))
    popularity = model.predict(song_features_poly.reshape(1,-1))
    return jsonify(popularity[0])

if __name__ == '__main__':
    app.run()
