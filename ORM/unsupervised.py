from routes import *
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||Unsupervised Kmeans/GaussianMixture
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def filter_by_race(race, list_of_games):
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
            if PCA_load.explained_variance_ratio_[0] > .30:
                PCA_list.append(PCA_load.components_[0])
        except:
            pass
    return PCA_list
    #return [joblib.load('PCA_Models/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib').components_[0] for gpid in filter_by_race_ if joblib.load('PCA_Models/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib').explained_variance_ratio_[0] > .75]

def construct_load_PCA(load_PCA_):
    return pd.DataFrame(load_PCA_)

def plot_PCA_vectors(construct_load_PCA_):
    pca = PCA(n_components = 4)
    tr = pca.fit_transform(construct_load_PCA_)
    explore_r4(tr[:,0], tr[:,1], tr[:,2], tr[:,3])
    return None

def plot_df_vectors(PCA_df, race):
    PCA_df = [game for game in PCA_df if game != None]
    if race == None:
        PCA_df_ = [df_list[player][0] for df_list in PCA_df for player in range(0,len(df_list))]
    else:
        PCA_df_ = [df_list[player][0] for df_list in PCA_df for player in range(0,len(df_list)) if df_list[player][3] == race]
    # dropped columns must occur in PCA_df function
    PCA_df_o = pd.concat(PCA_df_, sort = False).fillna(0).drop(columns = ['game_id', 'player_id', 'second'])
    import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    mm = MinMaxScaler()
    df_mm = mm.fit_transform(PCA_df_o)
    pca = PCA(n_components = 4)
    tr = pca.fit_transform(df_mm)
    explore_r4(tr[:,0], tr[:,1], tr[:,2], tr[:,3])

#B = plot_PCA_vectors(construct_load_PCA(load_PCA(filter_by_race(None, A), 'PSE')))

def plot_correlation_Heatmap(games):
    Terran = construct_load_PCA(load_PCA(filter_by_race('Terran', games)))
    Protoss = construct_load_PCA(load_PCA(filter_by_race('Protoss', games)))
    Zerg = construct_load_PCA(load_PCA(filter_by_race('Zerg', games)))

    df = pd.concat([Terran, Protoss, Zerg], axis = 0, sort = False)
    Y = df @ df.T
    Y = np.absolute(Y)
    #import pdb; pdb.set_trace()
    trace = go.Heatmap(z = np.array(Y))
    fig = go.Figure(data = [trace])
    offline.plot(fig)

def KMeans_(PCA_df ,n_clusters, n_init, name):
    km_ = KMeans(n_clusters = n_clusters, n_init = n_init)
    km_.fit(PCA_df)
    km_predict = km_.predict(PCA_df)
    joblib.dump(km_, 'KM_Models/' + name + '_' + str(n_clusters) + '.joblib')
    return pd.concat([PCA_df, pd.DataFrame(km_predict)], axis = 1, sort = False)

def GaussianMixture_(PCA_df, n_components, n_init, name):
    gm_ = GaussianMixture(n_components = n_components, n_init = n_init)
    gm_.fit(PCA_df)
    gm_predict = gm_.predict(PCA_df)
    joblib.dump(gm_, 'GM_Models/' + name + '_' + str(n_components) + '.joblib')
    return pd.concat([PCA_df,pd.DataFrame(gm_predict)], axis = 1, sort = False)

def plot_Unsupervised_Cluster(type_, n_, n_init, n_components, PCA_df, race, name, event_):
    if type_ == 'KMeans':
        model_ = np.array(KMeans_(construct_load_PCA(load_PCA(filter_by_race(race, PCA_df), event_)), n_, n_init, name))
    if type_ == 'GaussianMixture':
        model_ = np.array(GaussianMixture_(construct_load_PCA(load_PCA(filter_by_race(race, PCA_df), event_)), n_, n_init, name))
    pca_ = PCA(n_components = n_components)
    pca_fit_transform = pca_.fit_transform(model_[:,0:-2])
    explore_r4(pca_fit_transform[:,0], pca_fit_transform[:, 1], pca_fit_transform[:,2], model_[:,-1])
    return None
