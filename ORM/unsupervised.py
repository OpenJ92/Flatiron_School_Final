#from routes import *
import numpy as np
import sklearn
from PCA_ETL import *
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||Unsupervised Kmeans/GaussianMixture
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def filter_by_race(race, list_of_games):
    list_of_games = [game for game in list_of_games if isinstance(game,Game)]
    list_of_games = [game for game in list_of_games if len(game.players) == 2]
    if race == None:
        race_1 = [(game.id, game.players[0].id) for game in list_of_games]
        race_2 = [(game.id, game.players[1].id)  for game in list_of_games]
    else:
        race_1 = [(game.id, game.players[0].id) for game in list_of_games if game.playerOne_playrace == race]
        race_2 = [(game.id, game.players[1].id)  for game in list_of_games if game.playerTwo_playrace == race]
    return race_1 + race_2

def load_PCA(filter_by_race_, event_):
    filter_by_race_ = [id for id in filter_by_race_ if id != None]
    PCA_list = []
    for gpid in filter_by_race_:
        try:
            PCA_load = joblib.load('PCA_Models_' + event_ + '/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib')
            if PCA_load.explained_variance_ratio_[0] > 0:
                PCA_list.append(PCA_load.components_[0])
        except:
            pass
    return PCA_list
    #return [joblib.load('PCA_Models/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib').components_[0] for gpid in filter_by_race_ if joblib.load('PCA_Models/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib').explained_variance_ratio_[0] > .75]

def load_KMeans(race, event_):
    KM_list = []
    for file in [file for file in os.listdir('KM_Models') if race + '_' + event_ in file]:
        try:
            KM_list.append(('KM_Models/' + file, joblib.load('KM_Models/' + file)))
        except:
            pass
    return KM_list

def construct_load_PCA(load_PCA_):
    return pd.DataFrame(load_PCA_).fillna(0)

def normalize_PCA(construct_load_PCA_):
    construct_load_PCA_l = -1 * construct_load_PCA_[construct_load_PCA_ @ np.ones_like(construct_load_PCA_.iloc[0]) < 0]
    construct_load_PCA_u = construct_load_PCA_[construct_load_PCA_ @ np.ones_like(construct_load_PCA_.iloc[0]) >= 0]
    return pd.concat([construct_load_PCA_l, construct_load_PCA_u], sort = False)

def plot_PCA_vectors(construct_load_PCA_, normalize = True):
    if normalize:
        construct_load_PCA_ = normalize_PCA(construct_load_PCA_)
    pca = PCA(n_components = 4)
    tr = pca.fit_transform(construct_load_PCA_)
    explore_r4(tr[:,0], tr[:,1], tr[:,2], tr[:,3])
    return None

###Change this so it correlates by only Zerg, Terran and Protoss
def plot_correlation_Heatmap(games, event_, race_):
    df = normalize_PCA(construct_load_PCA(load_PCA(filter_by_race(race_, games), event_)))
    Y = df @ df.T
    trace = go.Heatmap(z = np.array(Y))
    fig = go.Figure(data = [trace])
    offline.plot(fig)

#Re-write unsupervised functions to load PCA objects
def KMeans_(PCA_df ,n_clusters, n_init, name):
    if PCA_df.shape[1] == 0:
        return None
    km_ = KMeans(n_clusters = n_clusters, n_init = n_init)
    km_.fit(PCA_df)
    km_predict = km_.predict(PCA_df)
    joblib.dump(km_, 'KM_Models/' + name + '_' + str(n_clusters)  + '_' + '.joblib')
    return pd.concat([PCA_df, pd.DataFrame(km_predict)], axis = 1, sort = False)

def GaussianMixture_(PCA_df, n_components, n_init, name):
    if PCA_df.shape[1] == 0:
        return None
    gm_ = GaussianMixture(n_components = n_components, n_init = n_init)
    gm_.fit(PCA_df)
    gm_predict = gm_.predict(PCA_df)
    joblib.dump(gm_, 'GM_Models/' + name + '_' + str(n_components) + '_' + '.joblib')
    return pd.concat([PCA_df,pd.DataFrame(gm_predict)], axis = 1, sort = False)

# def plot_Unsupervised_Cluster(type_, n_, n_init, n_components, PCA_df, race, name, event_, plot = False):
#     W = construct_load_PCA(load_PCA(filter_by_race(race, PCA_df), event_))
#     if type_ == 'KMeans':
#         model_ = np.array(KMeans_(W, n_, n_init, name))
#     if type_ == 'GaussianMixture':
#         model_ = np.array(GaussianMixture_(W, n_, n_init, name))
#     pca_ = PCA(n_components = n_components)
#     pca_fit_transform = pca_.fit_transform(model_[:,0:-2])
#     if plot:
#         explore_r4(pca_fit_transform[:,0], pca_fit_transform[:, 1], pca_fit_transform[:,2], model_[:,-1])
#     return None

A = filter_by_Game_highest_league(20)

def construct_Unsupervised_Models(func, num, event_):
    for race in ['Terran', 'Zerg', 'Protoss']:
        for n_components in list(range(2, num)):
            ##JUSTIFICATION for .fillna(0):  len of each singular vector whose elements were refilled resolve to one.
            df = normalize_PCA(construct_load_PCA(load_PCA(filter_by_race(race, A), event_))).fillna(0)
            func(df, n_components, 15, race + '_' + event_)

def silhouette_score_(race_, event_, list_of_games):
    ##JUSTIFICATION for .fillna(0):  len of each singular vector whose elements were refilled resolve to one.
    X = construct_load_PCA(load_PCA(filter_by_race(race_, list_of_games), event_)).fillna(0)
    list_KM = load_KMeans(race_, event_)
    #import pdb; pdb.set_trace()
    x = [int(KM[0].split('_')[-2]) for KM in list_KM]
    y = [silhouette_score(X, KM[1].labels_) for KM in list_KM]

    plt.scatter(x, y)
    plt.show()

def silhouette_samples_(race_, event_, list_of_games, model):
    pass

Plot game and PCA_vector

## Goal for rest of day:  Try to emulate http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
## plot_PCA_vectors(construct_load_PCA(load_PCA(filter_by_race(None, A), 'PSE')))
