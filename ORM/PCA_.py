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

# A, A_col = pipeline()

# query all professional replays and carry out kmeans and GaussianMixture on PCA
# vectors. Then ask the question of what exactly the unsupervised ML picked up on.

# pca = PCA()
# li = []
# #
# for i in range(0, len(B)):
#     for j in range(0, len(B[i])):
#         W = B[i][j][4].components_[0]
#         li.append(pd.DataFrame(W))
#
# li_A = np.array(pd.concat(li, axis = 1)).T
# E = pca.fit_transform(li_A.T)
# explore_r4(E[:,0], E[:,1], E[:,2], E[:,3])
# #Carry out this analysis on each other event class.
# n_clusters = 10
# km_ = KMeans(n_clusters = n_clusters, n_init = 20)
# km_fit = km_.fit(li_A)
# km_predict = km_.predict(li_A)

# n_clusters = 10
# gm_ = GaussianMixture(n_clusters = n_clusters, n_init = 20)
# gm_fit = gm_.fit(li_A)
# gm_predict = gm_.predict(li_A)

print('exit PCA')
