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
from sklearn.cluster import KMeans
import pickle
#%%
@op
def get_albums_data():
    df_albums = pd.read_csv('SpotGenTrack/Data Sources/spotify_albums.csv')

    return df_albums

#%%
@op
def save_labels_to_csv():

    df_artists = pd.read_csv('SpotGenTrack/Data Sources/spotify_artists.csv')
    n = 40

    df_artists['genres'] = df_artists['genres'].astype('string')
    df_artists = df_artists[df_artists['genres'] != '[]']
    df_artists['genres'] = df_artists['genres'].str.strip('[]').str.replace(' ','' ).str.replace("'", '')
    split_df = pd.DataFrame(df_artists.genres.str.split(",").tolist())
    labels = {}
    for i, name in enumerate(list(split_df[0].value_counts()[:n].index)):
        labels[name] = i
        
    pd.DataFrame.from_dict(labels, orient='index').to_csv('labels.csv',index=False)




#%%
@op
def get_artist_data():

    df_artists = pd.read_csv('SpotGenTrack/Data Sources/spotify_artists.csv')

    return df_artists

#%%
@op
def get_tracks_data():
    df_tracks = pd.read_csv('SpotGenTrack/Data Sources/spotify_tracks.csv')

    return df_tracks

#%%
@op
def get_joined_dataframes(df_tracks, df_artists, df_albums):

    df_join = df_tracks.set_index('id').join(df_artists.set_index('track_id'), on='id', lsuffix='_left', rsuffix='_right', how='inner')
    df_join = df_join.join(df_albums.set_index('track_id'), on=df_join.index, lsuffix='_left', rsuffix='_right', how = 'inner')


    return df_join

#%%

@op
def get_test_df(df_join):


    df_test = df_join[['acousticness', 'danceability', 'energy', 'instrumentalness','liveness','popularity','speechiness','tempo','valence']]

    return  df_test
#%%
@op
def get_clusters(df_test):
    scaler = StandardScaler()

    scaler.fit(df_test)

    scaled_data = scaler.transform(df_test)
    kmeans_model = KMeans(n_clusters=12)

    kmeans_model.fit(scaled_data)
    df_test['target'] = kmeans_model.labels_
    return df_test

#%%
@op
def save_df_to_csv(df_join, df_test):
    df_join.to_csv('Joined.csv',index=False)
    df_test.to_csv('test.csv',index=False)
    
#%%
@op
def normalize_and_train_knn(df_test):
    df_test.dropna(inplace=True)
    scaler = StandardScaler()
    scaler.fit(df_test.drop('target', axis=1))
    scaled_features = scaler.transform(df_test.drop('target',axis=1))
    df_test_feat = pd.DataFrame(scaled_features, columns = df_test.columns[:-1])
    y = df_test.iloc[:,-1].squeeze()
    X = df_test_feat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=20)

    knn.fit(X_train, y_train)
    
    return knn

# =============================================================================
# #%%
# @op
# def train_test_data (X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# 
#     return  X_train, X_test, y_train, y_test
# 
# #%%
# @op
# def knn_trainer(X_train, y_train, n):
#     knn = KNeighborsClassifier(n_neighbors=n)
# 
#     knn.fit(X_train, y_train)
# 
#     return knn
# =============================================================================

#%%
@op
def create_pickle(knn):
    knnPickle = open('knn_pickle_file', 'wb')
    pickle.dump(knn, knnPickle)
    knnPickle.close()
    
@job
def do_stuff():
    df_albums = get_albums_data()
    df_artists = get_artist_data()
    save_labels_to_csv()
    df_tracks = get_tracks_data()
    df_join = get_joined_dataframes(df_tracks, df_artists, df_albums)
    df_test = get_test_df(df_join)
    df_test = get_clusters(df_test)
    save_df_to_csv(df_join, df_test)
    knn = normalize_and_train_knn(df_test)
    create_pickle(knn)

