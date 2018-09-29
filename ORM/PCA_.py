#### Extract UnitBornEvents / UnitDonEvents by game.highest_league
#### and construct dataframe of aggregate actions and second.
#### Then carry out PCA and grab the first principle component.
#### This is a vector that describes our data. Then map to a hyper-sphere
#### of radius R (?) and carry out Kmeans on full set. This, I believe is a
#### reasonable measure of game stratagy.

### Goal for tonight. Go through professional Replays and carry out TruncatedSVD
### on each game. Use first principle component to carry out classificationself.
### Show that with 3 classes, one should find individual races.

##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT
##############FAILED EXPERIMENT

print('enter PCA')

from routes import *
import numpy as np
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()

# Make this a function that takes in a game, pulls events from said game then
# stores the principle vectors as a triplet. ie. (game_id, player_id, PCA)

def explore_r4(x,y,z,u):
     traces = go.Scatter3d(x = np.array(x), y = np.array(y), z = np.array(z), mode='markers', marker=dict(size = 3, color = u, colorscale = 'Jet', opacity = 0))
     fig = go.Figure(data=[traces])
     offline.plot(fig)
     return None

def PCA_UBE_df(game):
    #expected input -> <Game ( player )>
    game_events = game.events_UBE
    game_players = game.players
    game_id = game.id
    try:
        df_PlayerOne = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[0].id]).sort_values(by = ['second'])
        df_PlayerTwo = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[1].id]).sort_values(by = ['second'])
    except:
        print('Broken game: ' + str(game))
        return None
    df_PlayerOne_ = pd.get_dummies(df_PlayerOne['unit_type_name']).cumsum()
    df_PlayerTwo_ = pd.get_dummies(df_PlayerTwo['unit_type_name']).cumsum()
    df_PlayerOne_o = pd.concat([df_PlayerOne_, df_PlayerOne[['game_id', 'player_id']]], axis = 1, sort = False)
    df_PlayerTwo_o = pd.concat([df_PlayerTwo_, df_PlayerTwo[['game_id', 'player_id']]], axis = 1, sort = False)
    #expected output -> [[df, player_id, game_id], [df, player_id, game_id]]
    return [df_PlayerOne_o, game_players[0].id, game_id], [df_PlayerTwo_o, game_players[1].id, game_id]

def PCA_UBE_opperation(events_PCA):
    #expected input -> [[df, player_id, game_id], [df, player_id, game_id], [df, player_id, game_id], ..., [df, player_id, game_id]]
    events_PCA = [game for game in events_PCA if game != None]
    events_PCA_w_PC = []
    events_PCA_conglomerate = []

    for game in events_PCA:
        for player in game:
            events_PCA_conglomerate.append(player[0])

    events_PCA_conglomerate_df = pd.concat(events_PCA_conglomerate, axis = 0, sort = False).fillna(int(0))

    for game in range(0,len(events_PCA)):
        for player in range(0,len(events_PCA[game])):
            events_PCA[game][player].append(events_PCA_conglomerate_df[(events_PCA_conglomerate_df['player_id'] == events_PCA[game][player][1]) & (events_PCA_conglomerate_df['game_id'] == events_PCA[game][player][2])].drop(columns = list(events_PCA_conglomerate_df.columns[:15])))
            #####Remember to put back 'player_id' and 'game_id'

    for game in range(0,len(events_PCA)):
        for player in range(0,len(events_PCA[game])):
            df = events_PCA[game][player][3]
            mm = MinMaxScaler()
            df_mm = mm.fit_transform(df)
            _PCA = PCA()
            _PCA.fit(df_mm)
            ## 'player_id'_'game_id'.joblib
            joblib.dump(_PCA, 'PCA_Models_2/' + str(events_PCA[game][player][1]) + '_' + str(events_PCA[game][player][2]) + '.joblib')
            events_PCA[game][player].append(_PCA)

    #expected output -> [[[df, player_id, game_id, df_A, PCAobject], [df, player_id, game_id, df_A, PCAobject]], [[df, player_id, game_id, df_A, PCAobject], ..., [df, player_id, game_id, df_A, PCAobject]]
    return events_PCA, events_PCA[0][0][3].columns

def pipeline(func = None, parameters = None):
    if parameters == None:
        games = query()
    else:
        games = func(parameters)
    A = [PCA_UBE_df(game) for game in games]
    B = PCA_UBE_opperation(A)
    return B

def filter_by_race(race, list_of_games):
    list_of_games = [game for game in list_of_games if len(game.players) == 2]
    if race == None:
        race_1 = [(game.id, game.players[0].id) for game in list_of_games]
        race_2 = [(game.id, game.players[1].id)  for game in list_of_games]
    else:
        race_1 = [(game.id, game.players[0].id) for game in list_of_games if game.playerOne_playrace == race]
        race_2 = [(game.id, game.players[1].id)  for game in list_of_games if game.playerTwo_playrace == race]
    return race_1 + race_2

def load_PCA(filter_by_race_):
    filter_by_race_ = [id for id in filter_by_race_ if id != None]
    PCA_list = []
    for gpid in filter_by_race_:
        try:
            PCA_load = joblib.load('PCA_Models/' + str(gpid[1]) + '_' + str(gpid[0]) + '.joblib')
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


#B = plot_PCA_vectors(construct_load_PCA(load_PCA(filter_by_race('Zerg', A))))

def plot_correlation_Heatmap():
    Terran = construct_load_PCA(load_PCA(filter_by_race('Terran', A)))
    Protoss = construct_load_PCA(load_PCA(filter_by_race('Protoss', A)))
    Zerg = construct_load_PCA(load_PCA(filter_by_race('Zerg', A)))

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

def plot_Unsupervised_Cluster(type_, n_, n_init, n_components, PCA_df, race, name):
    if type_ == 'KMeans':
        model_ = np.array(KMeans_(construct_load_PCA(load_PCA(filter_by_race(race, PCA_df))), n_, n_init, name))
    if type_ == 'GaussianMixture':
        model_ = np.array(GaussianMixture_(construct_load_PCA(load_PCA(filter_by_race(race, PCA_df))), n_, n_init, name))
    pca_ = PCA(n_components = n_components)
    pca_fit_transform = pca_.fit_transform(model_[:,0:-2])
    explore_r4(pca_fit_transform[:,0], pca_fit_transform[:, 1], pca_fit_transform[:,2], model_[:,-1])
    return None

#B, A_col = pipeline()
A = query().all()
#A = filter_by_Game_highest_league(20)
plot_correlation_Heatmap()
E = KMeans_(construct_load_PCA(load_PCA(filter_by_race('Terran', A))), 50, 5, 'Terran_Professional')
F = GaussianMixture_(construct_load_PCA(load_PCA(filter_by_race('Terran', A))), 50, 5, 'Terran_Professional')

#plan for sept_28_18 --
##### Carry out Kmeans and Gaussian Mixture on PCA Data

print('exit PCA')
