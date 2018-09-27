import sklearn
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
import os

def SMOTE_(X_train, y_train):
    SMOTE = pd.concat([X_train.reset_index(), y_train.reset_index()], axis = 1)
    SMOTE_list = [SMOTE[SMOTE[col] == 1] for col in y_train.columns]

    max_sample = max(list(map(lambda x: len(x), SMOTE_list)))
    SMOTE_list_synthetic = []

    for i in SMOTE_list:
        synthetic_data = []
        for j in range(len(i), max_sample):
            a = i.iloc[0]
            b = i.iloc[np.random.randint(len(i))]
            synthetic_data.append(np.array(a + (np.random.random_sample()*(b - a))))
        try:
            t = pd.DataFrame(synthetic_data)
            t.columns = i.columns
            SMOTE_list_synthetic.append(pd.concat([i,t]))
        except:
            SMOTE_list_synthetic.append(i)

    SMOTE_complete = pd.concat(SMOTE_list_synthetic)
    SMOTE_complete_X_train = SMOTE_complete[X_train.columns]
    SMOTE_complete_y_train = SMOTE_complete[y_train.columns]

    return SMOTE_complete_X_train, SMOTE_complete_y_train

def explore_r4(x,y,z,w):
     traces = go.Scatter3d(x = np.array(x), y = np.array(y), z = np.array(z), mode='markers', marker=dict(size = 5, color = w, colorscale = 'Jet', opacity = 0))
     fig = go.Figure(data=[traces])
     offline.plot(fig)
     return None

def nn__(X_train_val, X_train, y_train_val, y_train, X_test, y_test, activation, epochs, batch, name, nodes = [100, 50], h_layers = 2, plot = True, dropout = [True,True,True,True]):

        nn_ = Sequential()

        nn_.add(Dense(X_train.shape[1], input_shape = (X_train.shape[1], ), activation = activation))

        if h_layers >= 1:
            nn_.add(Dense(nodes[0], activation = activation))
            if dropout[0]:
                nn_.add(Dropout(.2))
        if h_layers >= 2:
            nn_.add(Dense(nodes[1], activation = activation))
            if dropout[1]:
                nn_.add(Dropout(.2))
        if h_layers >= 3:
            nn_.add(Dense(nodes[2], activation = activation))
            if dropout[2]:
                nn_.add(Dropout(.2))
        if h_layers >= 4:
            nn_.add(Dense(nodes[3], activation = activation))
            if dropout[3]:
                nn_.add(Dropout(.2))

        nn_.add(Dense(y_train.shape[1], activation = 'softmax'))

        nn_.summary()
        nn_.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        nn_.fit(X_train, y_train, batch_size = batch, epochs = epochs, verbose = 1, validation_data = (X_train_val, y_train_val))

        if plot == plot:
            model_val_dict = nn_.history.history
            loss_values = model_val_dict['loss']
            val_loss_values = model_val_dict['val_loss']
            acc_values = model_val_dict['acc']
            val_acc_values = model_val_dict['val_acc']

            epochs_ = range(1, len(loss_values) + 1)
            plt.plot(epochs_, loss_values, 'g', label='Training loss')
            plt.plot(epochs_, val_loss_values, 'g.', label='Validation loss')
            plt.plot(epochs_, acc_values, 'r', label='Training acc')
            plt.plot(epochs_, val_acc_values, 'r.', label='Validation acc')

            plt.title('Training & validation loss / accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        nn_.save('./models_/nn_/' + name + '_' + activation +'_' + str(epochs) + '_' + str(batch) + '_' + str(h_layers) + '_' + '_'.join([str(i) for i in nodes]) + '_' + str(dropout[0]) + '.h5')
        print(nn_.evaluate(X_test, y_test))
        print(confusion_matrix(nn_.predict_classes(X_test), np.array(y_test).argmax(axis = 1)))
        print(classification_report(nn_.predict_classes(X_test), np.array(y_test).argmax(axis = 1)))
        return nn_

def cnn__(X_train_val, X_train, y_train_val, y_train, X_test, y_test, activation, epochs, batch, name, nodes = [100, 50], h_layers = 2, plot = True, dropout = [True,True,True,True]):

        cnn_ = Sequential()

        cnn_.add(Dense(X_train.shape[1], input_shape = (X_train.shape[1], ), activation = activation))

        if h_layers >= 1:
            cnn_.add(Dense(nodes[0], activation = activation))
            if dropout[0]:
                cnn_.add(Dropout(.2))
        if h_layers >= 2:
            cnn_.add(Dense(nodes[1], activation = activation))
            if dropout[1]:
                cnn_.add(Dropout(.2))
        if h_layers >= 3:
            cnn_.add(Dense(nodes[2], activation = activation))
            if dropout[2]:
                cnn_.add(Dropout(.2))
        if h_layers >= 4:
            cnn_.add(Dense(nodes[3], activation = activation))
            if dropout[3]:
                cnn_.add(Dropout(.2))

        cnn_.add(Dense(y_train.shape[1], activation = 'softmax'))

        cnn_.summary()
        cnn_.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
        cnn_.fit(X_train, y_train, batch_size = batch, epochs = epochs, verbose = 1, validation_data = (X_train_val, y_train_val))

        if plot == plot:
            model_val_dict = cnn_.history.history
            loss_values = model_val_dict['loss']
            val_loss_values = model_val_dict['val_loss']
            acc_values = model_val_dict['acc']
            val_acc_values = model_val_dict['val_acc']

            epochs_ = range(1, len(loss_values) + 1)
            plt.plot(epochs_, loss_values, 'g', label='Training loss')
            plt.plot(epochs_, val_loss_values, 'g.', label='Validation loss')
            plt.plot(epochs_, acc_values, 'r', label='Training acc')
            plt.plot(epochs_, val_acc_values, 'r.', label='Validation acc')

            plt.title('Training & validation loss / accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        cnn_.save('./models_/cnn_/' + name + '_' + activation +'_' + str(epochs) + '_' + str(batch) + '_' + str(h_layers) + '_' + '_'.join([str(i) for i in nodes]) + '_' + str(dropout[0]) + '.h5')
        print(cnn_.evaluate(X_test, y_test))
        print(confusion_matrix(cnn_.predict_classes(X_test), np.array(y_test).argmax(axis = 1)))
        print(classification_report(cnn_.predict_classes(X_test), np.array(y_test).argmax(axis = 1)))
        return nn_

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Goals
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# - Carry out Unsupervised stats on PCA data as a means to figure out what exactly
#       branches are. (Look back to Statistics Module for direction) - use information
#       to potentially train several different models for each branch.
# - Construct Neural Net to predict player action (BCE / TPC) given game state
#       Note: TPC events have coordinates - make inhibitor fuction that prevents
#               bad decisions by the attack bot. Note that Attack Move depends on map state
# - Construct Random Forest predict player action (BCE / TPC) given game state
#       Note: TPC events have coordinates - make inhibitor fuction that prevents
#               bad decisions by the attack bot. Note that Attack Move depends on map state
# - Use Kmeans to group data into classes

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||NN - BCE - TPC
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    ZvT_BCE = pd.read_csv('./matchup_/ZvT_BCE.csv')
    # PvT_BCE = pd.read_csv('./matchup_/PvT_BCE.csv')
    # TvT_BCE = pd.read_csv('./matchup_/TvT_BCE.csv')
    ZvT_TPC = pd.read_csv('./matchup_/ZvT_TPC.csv')
    # PvT_TPC = pd.read_csv('./matchup_/PvT_TPC.csv')
    # TvT_TPC = pd.read_csv('./matchup_/TvT_TPC.csv')

    BCE_ = ZvT_BCE
    TPC_ = ZvT_TPC
    # BCE_ = pd.concat([ZvT_BCE,PvT_BCE,TvT_BCE], axis = 0)
    # TPC_ = pd.concat([ZvT_TPC,PvT_TPC,TvT_TPC], axis = 0)

    BCE_ = BCE_[(BCE_['BCE_ability_name'] == 'TrainCyclone') | (BCE_['BCE_ability_name'] == 'TrainMarine') |
                (BCE_['BCE_ability_name'] == 'TrainMedivac') | (BCE_['BCE_ability_name'] == 'BuildSiegeTank') |
                (BCE_['BCE_ability_name'] == 'TrainBanshee') | (BCE_['BCE_ability_name'] == 'TrainMarauder')]

    TPC_ = TPC_[(TPC_['TPC_ability_name'] == 'BuildBarracks') |
                (TPC_['TPC_ability_name'] == 'BuildCommandCenter') | (TPC_['TPC_ability_name'] == 'BuildEngineeringBay') |
                (TPC_['TPC_ability_name'] == 'BuildFactory') | (TPC_['TPC_ability_name'] == 'BuildStarport')]

    BCE_X = BCE_.drop(columns = ['Unnamed: 0', 'BCE_ability_name', 'UDE_map_name', 'BCE_player', 'UDE_winner', 'PSE_food_made',
                                'PSE_food_used', 'PSE_minerals_collection_rate', 'PSE_minerals_current', 'PSE_vespene_collection_rate',
                                'PSE_vespene_current', 'PSE_workers_active_count',
                                'UBE_utn_Terran_Reaper', 'UBE_utn_Terran_WidowMine', 'UDE_utcsi_Terran_Bunker',
                                'UDE_utcsi_Terran_OrbitalCommand'])
    TPC_X = TPC_.drop(columns = ['Unnamed: 0', 'TPC_ability_name', 'TPC_x', 'TPC_y', 'UDE_map_name', 'TPC_player', 'UDE_winner',
                                'PSE_food_made', 'PSE_food_used', 'PSE_minerals_collection_rate', 'PSE_minerals_current',
                                'PSE_vespene_collection_rate', 'PSE_vespene_current', 'PSE_workers_active_count',
                                'UBE_utn_Terran_Reaper', 'UBE_utn_Terran_WidowMine', 'UDE_utcsi_Terran_Bunker',
                                'UDE_utcsi_Terran_OrbitalCommand'])
    BCE_y = pd.get_dummies(BCE_['BCE_ability_name'])
    TPC_y = pd.get_dummies(TPC_['TPC_ability_name'])

    BCE_X_col = BCE_X.columns
    BCE_y_col = BCE_y.columns
    TPC_X_col = TPC_X.columns
    TPC_y_col = TPC_y.columns

    SSc = StandardScaler()

    BCE_X = SSc.fit_transform(BCE_X)
    TPC_X = SSc.fit_transform(TPC_X)

    BCE_X_train, BCE_X_test, BCE_y_train, BCE_y_test = train_test_split(BCE_X, BCE_y, test_size=0.4, random_state=42, shuffle = True, stratify = BCE_['BCE_ability_name'])
    TPC_X_train, TPC_X_test, TPC_y_train, TPC_y_test = train_test_split(TPC_X, TPC_y, test_size=0.4, random_state=42, shuffle = True, stratify = TPC_['TPC_ability_name'])

    BCE_X_train = pd.DataFrame(BCE_X_train, columns = BCE_X_col)
    BCE_y_train = pd.DataFrame(BCE_y_train, columns = BCE_y_col)
    TPC_X_train = pd.DataFrame(TPC_X_train, columns = TPC_X_col)
    TPC_y_train = pd.DataFrame(TPC_y_train, columns = TPC_y_col)

    BCE_X_train, BCE_y_train = SMOTE_(BCE_X_train, BCE_y_train)
    #TPC_X_train, TPC_y_train = SMOTE_(TPC_X_train, TPC_y_train)

    BCE_X_train_val, BCE_X_train, BCE_y_train_val, BCE_y_train = train_test_split(BCE_X_train, BCE_y_train, test_size=0.8, random_state=42, shuffle = True)
    TPC_X_train_val, TPC_X_train, TPC_y_train_val, TPC_y_train = train_test_split(TPC_X_train, TPC_y_train, test_size=0.8, random_state=42, shuffle = True)

    nn_ = nn__(BCE_X_train_val, BCE_X_train, BCE_y_train_val, BCE_y_train, BCE_X_test, BCE_y_test, 'relu', 20, 10000,'BCE_', h_layers = 4, nodes = [250,500,250,125])

    ##rewrite the preceeding as a function. The saved file should go to a folder of nn
    ##and whose name are the inputs of the function and time.
            ##Create for loop itterating through several different variations.
    ##Try next -- divide dataset into matchup. ie - TvZ, TvT, TvP perhaps by map as well?
    ##Carry out a 'GridSearch'

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Kmeans
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if True:
    ZvT_BCE = pd.read_csv('./matchup_/ZvT_BCE_c.csv')
    PvT_BCE = pd.read_csv('./matchup_/PvT_BCE_c.csv')
    TvT_BCE = pd.read_csv('./matchup_/TvT_BCE_c.csv')
    ZvT_TPC = pd.read_csv('./matchup_/ZvT_TPC_c.csv')
    PvT_TPC = pd.read_csv('./matchup_/PvT_TPC_c.csv')
    TvT_TPC = pd.read_csv('./matchup_/TvT_TPC_c.csv')

    BCE_ = pd.concat([ZvT_BCE,PvT_BCE,TvT_BCE], axis = 0)
    TPC_ = pd.concat([ZvT_TPC,PvT_TPC,TvT_TPC], axis = 0)

    BCE_X = BCE_.select_dtypes(exclude = 'object').drop(columns = ['Unnamed: 0'])
    BCE_X = BCE_X[[i for i in BCE_X.columns if 'Terran' in i]].fillna(0)
    BCE_X = BCE_X.drop(columns = ['UBE_utn_Terran_AutoTurret','UBE_utn_Terran_BeaconArmy',
       'UBE_utn_Terran_BeaconAttack', 'UBE_utn_Terran_BeaconAuto',
       'UBE_utn_Terran_BeaconClaim', 'UBE_utn_Terran_BeaconCustom1',
       'UBE_utn_Terran_BeaconCustom2', 'UBE_utn_Terran_BeaconCustom3',
       'UBE_utn_Terran_BeaconCustom4', 'UBE_utn_Terran_BeaconDefend',
       'UBE_utn_Terran_BeaconDetect', 'UBE_utn_Terran_BeaconExpand',
       'UBE_utn_Terran_BeaconHarass', 'UBE_utn_Terran_BeaconIdle',
       'UBE_utn_Terran_BeaconRally', 'UBE_utn_Terran_BeaconScout',
       'UBE_utn_Terran_ChangelingMarine', 'UBE_utn_Terran_Nuke',
       'UBE_utn_Terran_ChangelingMarineShield', 'UBE_utn_Terran_Ghost',
       'UBE_utn_Terran_GhostAlternate','UBE_utn_Terran_KD8Charge',
       'UBE_utn_Terran_MULE', 'UBE_utn_Terran_RavenRepairDrone',
       'UBE_utn_Zerg_InfestedTerransEgg','UDE_utcsi_Terran_BarracksFlying',
       'UDE_utcsi_Terran_CommandCenterFlying','UDE_utcsi_Terran_FactoryFlying',
       'UDE_utcsi_Terran_OrbitalCommand', 'UDE_utcsi_Terran_OrbitalCommandFlying',
       'UDE_utcsi_Terran_PlanetaryFortress', 'UDE_utcsi_Terran_Reactor',
       'UDE_utcsi_Terran_SensorTower', 'UDE_utcsi_Terran_StarportFlying',
       'UDE_utcsi_Terran_SupplyDepotLowered', 'UDE_utcsi_Terran_TechLab'])

    TPC_X = TPC_.select_dtypes(exclude = 'object').drop(columns = ['Unnamed: 0'])
    TPC_X = TPC_X[[i for i in TPC_X.columns if 'Terran' in i]].fillna(0)
    TPC_X = TPC_X.drop(columns = ['UBE_utn_Terran_AutoTurret','UBE_utn_Terran_BeaconArmy',
       'UBE_utn_Terran_BeaconAttack', 'UBE_utn_Terran_BeaconAuto',
       'UBE_utn_Terran_BeaconClaim', 'UBE_utn_Terran_BeaconCustom1',
       'UBE_utn_Terran_BeaconCustom2', 'UBE_utn_Terran_BeaconCustom3',
       'UBE_utn_Terran_BeaconCustom4', 'UBE_utn_Terran_BeaconDefend',
       'UBE_utn_Terran_BeaconDetect', 'UBE_utn_Terran_BeaconExpand',
       'UBE_utn_Terran_BeaconHarass', 'UBE_utn_Terran_BeaconIdle',
       'UBE_utn_Terran_BeaconRally', 'UBE_utn_Terran_BeaconScout',
       'UBE_utn_Terran_ChangelingMarine', 'UBE_utn_Terran_Nuke',
       'UBE_utn_Terran_ChangelingMarineShield', 'UBE_utn_Terran_Ghost',
       'UBE_utn_Terran_GhostAlternate','UBE_utn_Terran_KD8Charge',
       'UBE_utn_Terran_MULE', 'UBE_utn_Terran_RavenRepairDrone',
       'UBE_utn_Zerg_InfestedTerransEgg','UDE_utcsi_Terran_BarracksFlying',
       'UDE_utcsi_Terran_CommandCenterFlying','UDE_utcsi_Terran_FactoryFlying',
       'UDE_utcsi_Terran_OrbitalCommand', 'UDE_utcsi_Terran_OrbitalCommandFlying',
       'UDE_utcsi_Terran_PlanetaryFortress', 'UDE_utcsi_Terran_Reactor',
       'UDE_utcsi_Terran_SensorTower', 'UDE_utcsi_Terran_StarportFlying',
       'UDE_utcsi_Terran_SupplyDepotLowered', 'UDE_utcsi_Terran_TechLab'])

    n_clusters = 3

    km_ = KMeans(n_clusters = n_clusters, n_init = 20)
    ss_ = StandardScaler()
    pca = PCA()

    BCE_X_s = ss_.fit_transform(BCE_X)
    BCE_X_s = pd.DataFrame(BCE_X_s, columns = BCE_X.columns)
    BCE_X_s_ = BCE_X_s[(np.sqrt(sum([BCE_X_s[col]**2 for col in BCE_X_s.columns])) > 5) & (np.sqrt(sum([BCE_X_s[col]**2 for col in BCE_X_s.columns])) < 15)]
    BCE_X_s_km = km_.fit(BCE_X_s_)
    BCE_X_s_km_p = km_.predict(BCE_X_s_)
    BCE_X_s_p = pca.fit_transform(BCE_X_s_)

    if True:
        plt.scatter(range(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_.cumsum())
        plt.show()

    if True:
        explore_r4(BCE_X_s_p[:,0],BCE_X_s_p[:,2],BCE_X_s_p[:,3],BCE_X_s_km_p)
        W = pd.concat([BCE_X_s, pd.DataFrame(BCE_X_s_km_p, columns = ['KMeans_' + str(n_clusters)])], axis = 1)
        c = [W[W['KMeans_' + str(n_clusters)] == i] for i in range(0,n_clusters)]

        for col in W.columns:
            for element in range(0,len(c)):
                plt.scatter(c[element][col], (element)*np.ones_like(c[element][col]), label = col + 'c' + str(element))
            plt.legend()
            plt.show()

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||Random Forest Classifier
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    pass

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||PCA
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    SSc = StandardScaler()
    pca = PCA()
    Q = BCE_.select_dtypes(exclude = 'object')
    Q = Q.drop(columns = ['Unnamed: 0'])
    Q_s = SSc.fit_transform(Q)
    Q_s_p = pca.fit_transform(Q_s)

    plt.scatter(range(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_.cumsum())
    plt.show()

    explore_r4(Q_s_p[:,0],Q_s_p[:,2],Q_s_p[:,3],Q_s_p[:,4])
