#%%
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

#%%
knn = pickle.load(open('knn_pickle_file', 'rb'))
def load_and_transform_df():
    data = pd.read_csv('Joined.csv')
    data2 = data[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'popularity', 'speechiness',
                  'tempo', 'valence', 'name']]
    scaler = StandardScaler()
    scaler.fit(data2.drop('name', axis=1))
    scaled_features = scaler.transform(data2.drop('name',axis=1))
    scaled_features = pd.DataFrame(scaled_features)
    scaled_features['name'] = data2['name']
    scaled_features.columns = data2.columns
    return scaled_features, data

def load_table(scaled_features):
    name = scaled_features['name'].drop_duplicates()
    return name


def find_songs(scaled_features, choice):
    #if there is a Duplicated song with different Attributes, choose the first one (.drop_duplicates()[:1])
    y = scaled_features[scaled_features['name'] == choice][['acousticness', 'danceability', 'energy',
                                                            'instrumentalness','liveness','popularity','speechiness','tempo',
                                                            'valence']].drop_duplicates()[:1].squeeze().to_numpy().reshape(1,-1)
    l = knn.kneighbors(y, n_neighbors=6, return_distance=False)
    return l

scaled_features, data = load_and_transform_df()
name = load_table(scaled_features)


#%%
st.title('Welcome to find your Song APP')
st.text('Chose a song from to Input on the right. The App will calculate six related songs and present them to you')
#%%

choice = st.sidebar.selectbox('Select your vehicle:', name)
#%%
l = find_songs(scaled_features, choice)

df =data.filter(list(l[0]), axis=0)[['name', 'acousticness', 'danceability', 'energy', 'instrumentalness','liveness',
                                     'popularity','speechiness','tempo','valence']]
#%%
st.table(df)
#%%
