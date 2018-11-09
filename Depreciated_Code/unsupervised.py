#from routes import *
import numpy as np
import sklearn
from PCA_ETL import *
import os
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go
from plotly import tools
offline.init_notebook_mode()

# GOAL FOR MONDAY -- Begin to rewrite file for new schema

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
            if PCA_load.explained_variance_ratio_[0] > .7:
                PCA_list.append(PCA_load.components_[0])
        except:
            pass
    return PCA_list
    #return [joblib.load('PCA_Models/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib').components_[0] for gpid in filter_by_race_ if joblib.load('PCA_Models/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib').explained_variance_ratio_[0] > .75]

def load_KMeans(race, event_):
    KM_list = []
    if race == None:
        for file in [file for file in os.listdir('KM_Models') if event_ in file]:
            try:
                KM_list.append(('KM_Models/' + file, joblib.load('KM_Models/' + file)))
            except:
                pass
    else:
        for file in [file for file in os.listdir('KM_Models') if race + '_' + event_ in file]:
            try:
                KM_list.append(('KM_Models/' + file, joblib.load('KM_Models/' + file)))
            except:
                pass
    return KM_list

def load_GaussianMixture(race, event_):
    GM_list = []
    if race == None:
        for file in [file for file in os.listdir('GM_Models') if event_ in file]:
            try:
                GM_list.append(('GM_Models/' + file, joblib.load('GM_Models/' + file)))
            except:
                pass
    else:
        for file in [file for file in os.listdir('GM_Models') if race + '_' + event_ in file]:
            try:
                GM_list.append(('GM_Models/' + file, joblib.load('GM_Models/' + file)))
            except:
                pass
    return GM_list

def construct_load_PCA(load_PCA_):
    return pd.DataFrame(load_PCA_).fillna(0)

def normalize_PCA(construct_load_PCA_):
    construct_load_PCA_l = -1 * construct_load_PCA_[construct_load_PCA_ @ np.ones_like(construct_load_PCA_.iloc[0]) < 0]
    construct_load_PCA_u = construct_load_PCA_[construct_load_PCA_ @ np.ones_like(construct_load_PCA_.iloc[0]) >= 0]
    return pd.concat([construct_load_PCA_l, construct_load_PCA_u], sort = False)

def plot_PCA_vectors(construct_load_PCA_, name, normalize = True):
    if normalize:
        construct_load_PCA_ = normalize_PCA(construct_load_PCA_)
    pca = PCA(n_components = 4)
    tr = pca.fit_transform(construct_load_PCA_)
    explore_r4(tr[:,0], tr[:,1], tr[:,2], tr[:,3], name)
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

def construct_Unsupervised_Models(func, num, event_):
    for race in ['Terran', 'Zerg', 'Protoss']:
        for n_components in list(range(2, num)):
            ##JUSTIFICATION for .fillna(0):  len of each singular vector whose elements were refilled resolve to one.
            df = normalize_PCA(construct_load_PCA(load_PCA(filter_by_race(race, A), event_))).fillna(0)
            func(df, n_components, 15, race + '_' + event_)

def silhouette_score_(race_, event_, list_of_games, load_func, plot = False):
    ##JUSTIFICATION for .fillna(0):  len of each singular vector whose elements were refilled resolve to one.
    X = normalize_PCA(construct_load_PCA(load_PCA(filter_by_race(race_, list_of_games), event_))).fillna(0)
    list_KM = load_func(race_, event_)
    x = [int(KM[0].split('_')[-2]) for KM in list_KM]
    y = [silhouette_score(X, KM[1].labels_) for KM in list_KM]
    if plot:
        plt.scatter(x, y)
        plt.show()
    return pd.concat([pd.DataFrame(x, columns = ['n_components']), pd.DataFrame(y, columns = ['silhouette_score'])], axis = 1, sort = False).sort_values(by = ['silhouette_score'])

def silhouette_samples_(race_, event_, list_of_games):
    X = normalize_PCA(construct_load_PCA(load_PCA(filter_by_race(race_, list_of_games), event_))).fillna(0)
    list_KM = load_KMeans(race_, event_)
    for KM in range(0,len(list_KM)):
        s_score = silhouette_score(X, list_KM[KM][1].labels_)
        s_samples = silhouette_samples(X, list_KM[KM][1].labels_)
        lin = np.linspace(0,1,len(s_samples))
        for samp in range(0,len(s_samples)):
            plt.plot([0 + KM, 1 + KM], [s_samples[samp], s_samples[samp]])

    plt.show()
    return None

A = filter_by_Game_highest_league(20)

if False:
    #
    # BCE = construct_Unsupervised_Models(GaussianMixture_, 10, 'BCE')
    # TPE = construct_Unsupervised_Models(GaussianMixture_, 10, 'TPE')
    # UBE = construct_Unsupervised_Models(GaussianMixture_, 10, 'UBE')
    # UDE = construct_Unsupervised_Models(GaussianMixture_, 10, 'UDE')
    # PSE = construct_Unsupervised_Models(GaussianMixture_, 10, 'PSE')

    # DESICION ON SILHOUETTE METRIC  --> (n_clusers, silhouette_score max)
    # --- BCE -- Conclusive
    Terran_BCE = silhouette_score_('Terran', 'BCE', A, load_GaussianMixture, plot = True) ## (13, 0.246829)
    Zerg_BCE = silhouette_score_('Zerg', 'BCE', A,load_GaussianMixture, plot = True) ## (6, 0.332655)
    Protoss_BCE = silhouette_score_('Protoss', 'BCE', A,load_GaussianMixture, plot = True) ## (8, 0.354308)

    # --- TPE -- Inconclusive
    Terran_TPE = silhouette_score_('Terran', 'TPE', A,load_GaussianMixture, plot = True) ## () -- Inconclusive
    Zerg_TPE = silhouette_score_('Zerg', 'TPE', A,load_GaussianMixture, plot = True) ## () -- Inconclusive
    Protoss_TPE = silhouette_score_('Protoss', 'TPE', A,load_GaussianMixture, plot = True) ## (9, 0.299635)

    # --- UBE -- Inconclusive
    Terran_UBE = silhouette_score_('Terran', 'UBE', A,load_GaussianMixture, plot = True) ## (12, 0.273744)
    Zerg_UBE = silhouette_score_('Zerg', 'UBE', A,load_GaussianMixture, plot = True) ## (8, 0.437116)
    Protoss_UBE = silhouette_score_('Protoss', 'UBE', A,load_GaussianMixture, plot = True) ## (14, 0.271800) -- Inconclusive

    # --- UDE -- Conclusive
    Terran_UDE = silhouette_score_('Terran', 'UDE', A,load_GaussianMixture, plot = True) ## (10, 0.229572)
    Zerg_UDE = silhouette_score_('Zerg', 'UDE', A,load_GaussianMixture, plot = True) ## (8, 0.233198)
    Protoss_UDE = silhouette_score_('Protoss', 'UDE', A,load_GaussianMixture, plot = True) ## (10, 0.258104)

    # --- PSE -- Inconclusive
    Terran_PSE = silhouette_score_('Terran', 'PSE', A, plot = True) ## () -- Inconclusive
    Zerg_PSE = silhouette_score_('Zerg', 'PSE', A, plot = True) ## () -- Inconclusive
    Protoss_PSE = silhouette_score_('Protoss', 'PSE', A, plot = True) ## () -- Inconclusive

def explore_(list_of_games, race_, event_, n_clusters):
    #Q dictionary will change
    Q = {'BCE': {'Terran': n_clusters, 'Zerg': n_clusters, 'Protoss': n_clusters, 'func': PCA_BCE_df},
         'UDE': {'Terran': n_clusters, 'Zerg': n_clusters, 'Protoss': n_clusters, 'func': PCA_UDE_df},
         'TPE': {'Terran': n_clusters, 'Zerg': n_clusters, 'Protoss': n_clusters, 'func': PCA_TPE_df},
         'UBE': {'Terran': n_clusters, 'Zerg': n_clusters, 'Protoss': n_clusters, 'func': PCA_UBE_df},
         'PSE': {'Terran': n_clusters, 'Zerg': n_clusters, 'Protoss': n_clusters, 'func': PCA_PSE_df}}
    KM = joblib.load('KM_Models/' + race_ + '_' + event_ + '_' + Q[event_][race_] + '_.joblib')
    # W = pd.concat([Q[event_]['func'](game)[player][0] for game in list_of_games for player in range(0,2) if Q[event_]['func'](game)[player][3] == race_], axis = 0, sort = False).fillna(0)
    list_ = []


    for game in list_of_games:
        if (game.playerOne_playrace == race_) or (game.playerTwo_playrace == race_):
            Q_data = Q[event_]['func'](game)
            if (game.playerOne_playrace == race_):
                try:
                    Q_data_p1 = Q_data[0][0]
                    PCA_P1 = joblib.load('PCA_Models_' + event_ + '/' + str(game.players[0].id) + '_' + str(game.id) + '.joblib')
                    PCA_KM_pred_p1 = KM.predict(PCA_P1.components_[0].reshape(-1, 1).T)
                    PCA_KM_pred_p1_df = pd.DataFrame([PCA_KM_pred_p1 for i in range(0, len(Q_data_p1))], columns = ['cluster'])
                    P1_df = pd.concat([Q_data_p1, PCA_KM_pred_p1_df], axis = 1)
                    #import pdb; pdb.set_trace()
                    list_.append(P1_df)
                except:
                    pass

            if (game.playerTwo_playrace == race_):
                try:
                    Q_data_p2 = Q_data[1][0]
                    PCA_P2 = joblib.load('PCA_Models_' + event_ + '/' + str(game.players[1].id) + '_' + str(game.id) + '.joblib')
                    PCA_KM_pred_p2 = KM.predict(PCA_P2.components_[0].reshape(-1, 1).T)
                    PCA_KM_pred_p2_df = pd.DataFrame([PCA_KM_pred_p2 for i in range(0, len(Q_data_p2))], columns = ['cluster'])
                    P2_df = pd.concat([Q_data_p2, PCA_KM_pred_p2_df], axis = 1).iloc[-2:-1]
                    list_.append(P2_df)
                    #import pdb; pdb.set_trace()
                except:
                    pass

    #######Find way to augment this so that color represents cluster
    games_in_cluster = pd.concat(list_, sort = False, axis = 0).fillna(0).drop(columns = ['second', 'player_id', 'game_id'])
    go.Layout()
    #traces = [go.Box(games_in_cluster[games_in_cluster['cluster'] == element][col]) for col in games_in_cluster.columns for element in games_in_cluster['cluster'].unique()]
    traces = []
    for element in games_in_cluster['cluster'].unique():
        for col in games_in_cluster.columns:
            #import pdb; pdb.set_trace()
            traces.append(go.Box(y = games_in_cluster[games_in_cluster['cluster'] == element][col], name = col + '_' + str(element)))
    #look up how go.Box() works. Something bizzare is happening when our plot is resolving to well beyond the maximum of units/buildings constructed.
    fig = go.Figure(data = traces)
    offline.plot(fig)

    import pdb; pdb.set_trace()

def plot_Unsupervised_Cluster_PCA(race_, event_ ,n_clusters_, type_, games_):
    X = normalize_PCA((construct_load_PCA(load_PCA(filter_by_race(race_, games_), event_))))
    model = joblib.load(type_ + '_Models/' + race_ + '_' + event_ + '_' + str(n_clusters_) + '_.joblib')
    y = model.predict(X)
    pca = PCA(n_components = 4)
    X_t = pca.fit_transform(X)
    explore_r4(X_t[:,0], X_t[:,1], X_t[:,2], y, race_ + '_' + event_ + '_' + type_ + '_unsupervised')

def construct_Unsupervised_Cluster_PCA(games_):
    for type_ in ['KM', 'GM']:
        for race_ in ['Terran', 'Protoss', 'Zerg']:
            for event_ in ['PSE', 'UBE', 'UDE', 'TPE', 'BCE']:
                for n_clusters_ in range(0, 10):
                    plot_Unsupervised_Cluster_PCA(race_, event_ ,n_clusters_, type_, games_)

def plot_Unsupervised_Cluster_df():
    pass

import webbrowser

## Goal for rest of day:  Try to emulate http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
## plot_PCA_vectors(construct_load_PCA(load_PCA(filter_by_race(None, A), 'PSE')))
## plot_df_vectors(PCA_df, 'Terran')

## Look though silhouette_score_ results to specify number of clusters to use.
