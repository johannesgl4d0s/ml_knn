# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:21:31 2023

@author: johan
"""

from dagster import job, op
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
#%%
@op
def get_albums_data(path):
    df_albums = pd.read_csv(path)

    return df_albums



#%%
@op
def get_artist_data(path, n):
    """
    :param path: string to file
    :param n: int, top level of genres
    :return: dataframe, dictionary

    """
    df_artists = pd.read_csv(path)

    df_artists['genres'] = df_artists['genres'].astype('string')
    df_artists = df_artists[df_artists['genres'] != '[]']
    df_artists['genres'] = df_artists['genres'].str.strip('[]').str.replace(' ','' ).str.replace("'", '')
    split_df = pd.DataFrame(df_artists.genres.str.split(",").tolist())
    labels = {}
    for i, name in enumerate(list(split_df[0].value_counts()[:n].index)):
        labels[name] = i

    df_artists['genre_1'] = split_df[0]
    df_artists['target'] = df_artists['genre_1'].apply(lambda x: labels.get(x)).fillna(n)
    return df_artists, labels

#%%
@op
def get_tracks_data(path):
    df_tracks = pd.read_csv(path)

    return df_tracks

#%%
@op
def get_joined_dataframes(df_tracks, df_artists, df_albums):

    df_join = df_tracks.set_index('id').join(df_artists.set_index('track_id'), on='id', lsuffix='_left', rsuffix='_right', how='inner')
    df_join = df_join.join(df_albums.set_index('track_id'), on=df_join.index, lsuffix='_left', rsuffix='_right', how = 'inner')

    df_test = df_join[['acousticness', 'danceability', 'energy', 'instrumentalness','liveness','popularity','speechiness','tempo','valence','target']]

    return df_join, df_test

#%%
@op
def save_df_to_csv(df_join, df_test):
    df_join.to_csv('Joined.csv',index=False)
    df_test.to_csv('test.csv',index=False)
    
#%%
@op
def get_normalized_X_y(df_test):
    df_test.dropna(inplace=True)
    scaler = StandardScaler()
    scaler.fit(df_test.drop('target', axis=1))
    scaled_features = scaler.transform(df_test.drop('target',axis=1))
    df_test_feat = pd.DataFrame(scaled_features, columns = df_test.columns[:-1])
    y = df_test.iloc[:,-1].squeeze()
    X = df_test_feat
    return X, y

#%%
@op
def train_test_data (X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return  X_train, X_test, y_train, y_test

#%%
@op
def knn_trainer(X_train, y_train, n):
    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train, y_train)

    return knn

#%%
@op
def create_pickle(knn):
    knnPickle = open('knn_pickle_file', 'wb')
    pickle.dump(knn, knnPickle)
    knnPickle.close()
    
@job
def do_stuff():
    df_albums = get_albums_data(path = 'SpotGenTrack/Data Sources/spotify_albums.csv')
    df_artists, labels = get_artist_data(path='SpotGenTrack/Data Sources/spotify_artists.csv' , n = 40)
    df_tracks = get_tracks_data(path = 'SpotGenTrack/Data Sources/spotify_tracks.csv')
    df_join, df_test = get_joined_dataframes(df_tracks, df_artists, df_albums)
    save_df_to_csv(df_join, df_test)
    X,y = get_normalized_X_y(df_test)
    X_train, X_test, y_train, y_test = train_test_data(X,y)
    knn = knn_trainer(X_train, y_train, n= 7)
    create_pickle(knn)

