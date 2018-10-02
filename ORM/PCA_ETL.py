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

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||DataFrame Construction||Feature Selection
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

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

    if game.playerOne_playrace == 'Terran':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Banshee', 'CommandCenter', 'Cyclone', 'Marine', 'Medivac', 'Raven', 'Reaper',
                                                                                        'SiegeTank', 'VikingFighter', 'Hellion', 'Liberator', 'Thor', 'Marauder',
                                                                                        'WidowMine', 'HellionTank', 'Battlecruiser', 'Ghost']]]
    elif game.playerOne_playrace == 'Protoss':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Nexus', 'Stalker', 'Colossus', 'Disruptor', 'Immortal', 'WarpPrism', 'Observer',
                                                                                        'Adept', 'Phoenix', 'Oracle', 'Zealot', 'Sentry', 'Tempest', 'Carrier',
                                                                                        'VoidRay', 'Archon', 'Mothership', 'HighTemplar']]]
    else:
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Hatchery', 'Roach', 'Baneling',  'Mutalisk', 'Queen', 'Zergling', 'Corruptor', 'Hydralisk',
                                                                                        'Viper', 'Ultralisk']]]
    if game.playerTwo_playrace == 'Terran':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Banshee', 'CommandCenter', 'Cyclone', 'Marine', 'Medivac', 'Raven', 'Reaper',
                                                                                        'SiegeTank', 'VikingFighter', 'Hellion', 'Liberator', 'Thor', 'Marauder',
                                                                                        'WidowMine', 'HellionTank', 'Battlecruiser', 'Ghost']]]
    elif game.playerTwo_playrace == 'Protoss':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Nexus', 'Stalker', 'Colossus', 'Disruptor', 'Immortal', 'WarpPrism', 'Observer',
                                                                                        'Adept', 'Phoenix', 'Oracle', 'Zealot', 'Sentry', 'Tempest', 'Carrier',
                                                                                        'VoidRay', 'Archon', 'Mothership', 'HighTemplar']]]
    else:
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Hatchery', 'Roach', 'Baneling',  'Mutalisk', 'Queen', 'Zergling', 'Corruptor', 'Hydralisk',
                                                                                        'Viper', 'Ultralisk']]]

    df_PlayerOne_o = pd.concat([df_PlayerOne_, df_PlayerOne[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    df_PlayerTwo_o = pd.concat([df_PlayerTwo_, df_PlayerTwo[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    #expected output -> [[df, player_id, game_id], [df, player_id, game_id]]
    return [df_PlayerOne_o, game_players[0].id, game_id, game.playerOne_playrace], [df_PlayerTwo_o, game_players[1].id, game_id, game.playerTwo_playrace]

def PCA_PSE_df(game):
    #expected input -> <Game ( player )>
    game_events = game.events_PSE
    game_players = game.players
    game_id = game.id
    try:
        df_PlayerOne = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[0].id]).sort_values(by = ['second'])
        df_PlayerTwo = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[1].id]).sort_values(by = ['second'])
    except:
        print('Broken game: ' + str(game))
        return None
    df_PlayerOne_ = df_PlayerOne.drop(columns = ['name', 'player', 'player_id', 'game_id', 'player_id', '_sa_instance_state'])
    df_PlayerTwo_ = df_PlayerTwo.drop(columns = ['name', 'player', 'player_id', 'game_id', 'player_id', '_sa_instance_state'])
    df_PlayerOne_o = pd.concat([df_PlayerOne_, df_PlayerOne[['game_id', 'player_id']]], axis = 1, sort = False)
    df_PlayerTwo_o = pd.concat([df_PlayerTwo_, df_PlayerTwo[['game_id', 'player_id']]], axis = 1, sort = False)
    #expected output -> [[df, player_id, game_id], [df, player_id, game_id]]
    return [df_PlayerOne_o, game_players[0].id, game_id, game.playerOne_playrace], [df_PlayerTwo_o, game_players[1].id, game_id, game.playerTwo_playrace]

def PCA_TPE_df(game):
    #expected input -> <Game ( player )>
    game_events = game.events_TPE
    game_players = game.players
    game_id = game.id
    try:
        df_PlayerOne = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[0].id]).sort_values(by = ['second'])
        df_PlayerTwo = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[1].id]).sort_values(by = ['second'])
    except:
        print('Broken game: ' + str(game))
        return None

    df_PlayerOne_ = pd.get_dummies(df_PlayerOne['ability_name']).cumsum()
    df_PlayerTwo_ = pd.get_dummies(df_PlayerTwo['ability_name']).cumsum()
    # df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if 'Build' in col]]
    # df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if 'Build' in col]]
    if game.playerOne_playrace == 'Terran':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab', 'BuildCommandCenter',
                                        'BuildEngineeringBay', 'BuildFactory', 'BuildFactoryTechLab', 'BuildMissileTurret', 'BuildSensorTower', 'BuildStarport', 'BuildStarportReactor',
                                        'BuildStarportTechLab', 'BuildSupplyDepot', 'BuildFactoryReactor', 'BuildBunker', 'BuildFusionCore', 'BuildGhostAcademy']]]
    elif game.playerOne_playrace == 'Protoss':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['BuildCyberneticsCore', 'BuildDarkShrine', 'BuildForge', 'BuildGateway', 'BuildNexus', 'BuildPylon', 'BuildShieldBattery',
                                        'BuildTemplarArchive', 'BuildTwilightCouncil', 'BuildRoboticsBay', 'BuildRoboticsFacility', 'BuildStargate', 'BuildPhotonCannon', 'BuildFleetBeacon']]]
    else:
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['BuildHatchery', 'BuildRoachWarren', 'BuildSpawningPool', 'BuildBanelingNest', 'BuildCreepTumor', 'BuildEvolutionChamber',
                                        'BuildSpire', 'BuildSporeCrawler', 'BuildHydraliskDen', 'BuildInfestationPit', 'BuildSpineCrawler', 'BuildUltraliskCavern', 'BuildNydusNetwork', 'BuildLurkerDenMP']]]
    if game.playerTwo_playrace == 'Terran':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab', 'BuildCommandCenter',
                                        'BuildEngineeringBay', 'BuildFactory', 'BuildFactoryTechLab', 'BuildMissileTurret', 'BuildSensorTower', 'BuildStarport', 'BuildStarportReactor',
                                        'BuildStarportTechLab', 'BuildSupplyDepot', 'BuildFactoryReactor', 'BuildBunker', 'BuildFusionCore', 'BuildGhostAcademy']]]
    elif game.playerTwo_playrace == 'Protoss':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['BuildCyberneticsCore', 'BuildDarkShrine', 'BuildForge', 'BuildGateway', 'BuildNexus', 'BuildPylon', 'BuildShieldBattery',
                                        'BuildTemplarArchive', 'BuildTwilightCouncil', 'BuildRoboticsBay', 'BuildRoboticsFacility', 'BuildStargate', 'BuildPhotonCannon', 'BuildFleetBeacon']]]
    else:
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['BuildHatchery', 'BuildRoachWarren', 'BuildSpawningPool', 'BuildBanelingNest', 'BuildCreepTumor', 'BuildEvolutionChamber',
                                        'BuildSpire', 'BuildSporeCrawler', 'BuildHydraliskDen', 'BuildInfestationPit', 'BuildSpineCrawler', 'BuildUltraliskCavern', 'BuildNydusNetwork', 'BuildLurkerDenMP']]]

    df_PlayerOne_o = pd.concat([df_PlayerOne_, df_PlayerOne[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    df_PlayerTwo_o = pd.concat([df_PlayerTwo_, df_PlayerTwo[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    #expected output -> [[df, player_id, game_id], [df, player_id, game_id]]
    return [df_PlayerOne_o, game_players[0].id, game_id, game.playerOne_playrace], [df_PlayerTwo_o, game_players[1].id, game_id, game.playerTwo_playrace]

def PCA_UDE_df(game):
    #run function to figure exactly which columns are relevant to analysis.
    #expected input -> <Game ( player )>
    game_events = game.events_UDE
    game_players = game.players
    game_id = game.id
    try:
        df_PlayerOne = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[0].id]).sort_values(by = ['second'])
        df_PlayerTwo = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[1].id]).sort_values(by = ['second'])
    except:
        print('Broken game: ' + str(game))
        return None

    df_PlayerOne_ = pd.get_dummies(df_PlayerOne['unit']).cumsum()
    df_PlayerTwo_ = pd.get_dummies(df_PlayerTwo['unit']).cumsum()
    # df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if 'Build' in col]]
    # df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if 'Build' in col]]

    if game.playerOne_playrace == 'Terran':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Armory', 'Barracks', 'BarracksReactor', 'EngineeringBay', 'Factory',
                                        'FactoryTechLab', 'MissileTurret', 'OrbitalCommand', 'PlanetaryFortress', 'Refinery', 'SensorTower', 'Starport',
                                        'StarportReactor', 'StarportTechLab', 'BarracksTechLab', 'CommandCenter', 'FactoryReactor', 'Bunker',
                                        'FusionCore', 'GhostAcademy']]]
    elif game.playerOne_playrace == 'Protoss':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['CyberneticsCore', 'DarkShrine', 'Forge', 'Nexus', 'ShieldBattery',
                                        'TemplarArchive', 'TwilightCouncil', 'WarpGate','RoboticsBay', 'RoboticsFacility', 'Stargate', 'PhotonCannon', 'FleetBeacon',
                                        'Gateway']]]
    else:
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Hatchery', 'RoachWarren', 'SpawningPool', 'BanelingNest', 'EvolutionChamber',
                                        'Spire', 'GreaterSpire', 'HydraliskDen', 'InfestationPit', 'SpineCrawler', 'SporeCrawler', 'UltraliskCavern', 'Lair',
                                        'NydusNetwork', 'Hive', 'CreepTumorQueen', 'CreepTumor', 'LurkerDen']]]
    if game.playerTwo_playrace == 'Terran':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Armory', 'Barracks', 'BarracksReactor', 'EngineeringBay', 'Factory',
                                        'FactoryTechLab', 'MissileTurret', 'OrbitalCommand', 'PlanetaryFortress', 'Refinery', 'SensorTower', 'Starport',
                                        'StarportReactor', 'StarportTechLab', 'BarracksTechLab', 'CommandCenter', 'FactoryReactor', 'Bunker',
                                        'FusionCore', 'GhostAcademy']]]
    elif game.playerTwo_playrace == 'Protoss':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['CyberneticsCore', 'DarkShrine', 'Forge', 'Nexus', 'ShieldBattery',
                                        'TemplarArchive', 'TwilightCouncil', 'WarpGate','RoboticsBay', 'RoboticsFacility', 'Stargate', 'PhotonCannon', 'FleetBeacon',
                                        'Gateway']]]
    else:
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Hatchery', 'RoachWarren', 'SpawningPool', 'BanelingNest', 'EvolutionChamber',
                                        'Spire', 'GreaterSpire', 'HydraliskDen', 'InfestationPit', 'SpineCrawler', 'SporeCrawler', 'UltraliskCavern', 'Lair',
                                        'NydusNetwork', 'Hive', 'CreepTumorQueen', 'CreepTumor', 'LurkerDen']]]

    df_PlayerOne_o = pd.concat([df_PlayerOne_, df_PlayerOne[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    df_PlayerTwo_o = pd.concat([df_PlayerTwo_, df_PlayerTwo[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    #expected output -> [[df, player_id, game_id], [df, player_id, game_id]]
    return [df_PlayerOne_o, game_players[0].id, game_id, game.playerOne_playrace], [df_PlayerTwo_o, game_players[1].id, game_id, game.playerTwo_playrace]

def PCA_BCE_df(game):
    #run function to figure exactly which columns are relevant to analysis.
    #expected input -> <Game ( player )>
    game_events = game.events_BCE
    game_players = game.players
    game_id = game.id
    try:
        df_PlayerOne = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[0].id]).sort_values(by = ['second'])
        df_PlayerTwo = pd.DataFrame([vars(event) for event in game_events if event.player.id == game_players[1].id]).sort_values(by = ['second'])
    except:
        print('Broken game: ' + str(game))
        #import pdb; pdb.set_trace()
        return None

    df_PlayerOne_ = pd.get_dummies(df_PlayerOne['ability_name']).cumsum()
    df_PlayerTwo_ = pd.get_dummies(df_PlayerTwo['ability_name']).cumsum()
    # df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if ('Train' in col) or ('Build' in col)]]
    # df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if ('Train' in col) or ('Build' in col)]]

    if game.playerOne_playrace == 'Terran':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['BuildSiegeTank', 'TrainBanshee', 'TrainCyclone', 'TrainLiberator', 'TrainMarine', 'TrainMedivac',
                                        'TrainRaven', 'TrainReaper', 'TrainViking', 'BuildHellion', 'BuildThor', 'TrainMarauder', 'BuildWidowMine',
                                        'TrainBattlecruiser', 'TrainGhost', 'BuildBattleHellion', 'TrainNuke']]]
    elif game.playerOne_playrace == 'Protoss':
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['TrainStalker', 'TrainColossus', 'TrainDisruptor', 'TrainImmortal', 'TrainObserver', 'TrainWarpPrism', 'TrainAdept',
                                        'TrainZealot', 'TrainOracle', 'TrainPhoenix', 'TrainSentry', 'TrainTempest', 'TrainCarrier', 'TrainVoidRay', 'TrainMothership', 'TrainInterceptor']]]
    else:
        df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['MorphRoach', 'MorphToRavager', 'MorphMutalisk', 'MorphToOverseer', 'MorphZergling', 'TrainBaneling', 'TrainQueen',
                                        'MorphCorruptor', 'MorphHydralisk', 'MorphInfestor', 'MorphSwarmHost', 'MorphToBroodLord', 'MorphViper', 'MorphUltralisk']]]
    if game.playerTwo_playrace == 'Terran':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['BuildSiegeTank', 'TrainBanshee', 'TrainCyclone', 'TrainLiberator', 'TrainMarine', 'TrainMedivac',
                                        'TrainRaven', 'TrainReaper', 'TrainViking', 'BuildHellion', 'BuildThor', 'TrainMarauder', 'BuildWidowMine',
                                        'TrainBattlecruiser', 'TrainGhost', 'BuildBattleHellion', 'TrainNuke']]]
    elif game.playerTwo_playrace == 'Protoss':
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['TrainStalker', 'TrainColossus', 'TrainDisruptor', 'TrainImmortal', 'TrainObserver', 'TrainWarpPrism', 'TrainAdept',
                                        'TrainZealot', 'TrainOracle', 'TrainPhoenix', 'TrainSentry', 'TrainTempest', 'TrainCarrier', 'TrainVoidRay', 'TrainMothership', 'TrainInterceptor']]]
    else:
        df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['MorphRoach', 'MorphToRavager', 'MorphMutalisk', 'MorphToOverseer', 'MorphZergling', 'TrainBaneling', 'TrainQueen',
                                        'MorphCorruptor', 'MorphHydralisk', 'MorphInfestor', 'MorphSwarmHost', 'MorphToBroodLord', 'MorphViper', 'MorphUltralisk']]]

    #import pdb; pdb.set_trace()

    df_PlayerOne_o = pd.concat([df_PlayerOne_, df_PlayerOne[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    df_PlayerTwo_o = pd.concat([df_PlayerTwo_, df_PlayerTwo[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
    #expected output -> [[df, player_id, game_id], [df, player_id, game_id]]
    return [df_PlayerOne_o, game_players[0].id, game_id, game.playerOne_playrace], [df_PlayerTwo_o, game_players[1].id, game_id, game.playerTwo_playrace]

def PCA_operation(events_PCA, event_):
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
            events_PCA[game][player].append(events_PCA_conglomerate_df[(events_PCA_conglomerate_df['player_id'] == events_PCA[game][player][1]) & (events_PCA_conglomerate_df['game_id'] == events_PCA[game][player][2])])
            #####Remember to put back 'player_id' and 'game_id'

    for game in range(0,len(events_PCA)):
        for player in range(0,len(events_PCA[game])):
            df = events_PCA[game][player][0].drop(columns = ['game_id', 'player_id', 'second'])
            if df.shape[1] == 0:
                break
            mm = MinMaxScaler()
            df_mm = mm.fit_transform(df)
            _PCA = PCA()
            _PCA.fit(df_mm)
            ## 'player_id'_'game_id'.joblib
            joblib.dump(_PCA, 'PCA_Models_'+ event_ +'/' + str(events_PCA[game][player][1]) + '_' + str(events_PCA[game][player][2]) + '.joblib')
            events_PCA[game][player].append(_PCA)

    #expected output -> [[[df, player_id, game_id, df_A, PCAobject], [df, player_id, game_id, df_A, PCAobject]], [[df, player_id, game_id, df_A, PCAobject], ..., [df, player_id, game_id, df_A, PCAobject]]
    return events_PCA, events_PCA[0][0][0].columns

def TSVD_operation(events_PCA, event_):
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
            events_PCA[game][player].append(events_PCA_conglomerate_df[(events_PCA_conglomerate_df['player_id'] == events_PCA[game][player][1]) & (events_PCA_conglomerate_df['game_id'] == events_PCA[game][player][2])])
            #####Remember to put back 'player_id' and 'game_id'

    for game in range(0,len(events_PCA)):
        for player in range(0,len(events_PCA[game])):
            df = events_PCA[game][player][0].drop(columns = ['game_id', 'player_id', 'second'])
            mm = MinMaxScaler()
            df_mm = mm.fit_transform(df)
            _TruncatedSVD = TruncatedSVD()
            _TruncatedSVD.fit(df_mm)
            ## 'player_id'_'game_id'.joblib
            joblib.dump(_TruncatedSVD, 'TSVD_Models_'+ event_ +'/' + str(events_PCA[game][player][1]) + '_' + str(events_PCA[game][player][2]) + '.joblib')
            events_PCA[game][player].append(_TruncatedSVD)

    #expected output -> [[[df, player_id, game_id, df_A, PCAobject], [df, player_id, game_id, df_A, PCAobject]], [[df, player_id, game_id, df_A, PCAobject], ..., [df, player_id, game_id, df_A, PCAobject]]
    return events_PCA, events_PCA[0][0][0].columns

def plot_df_vectors(PCA_df, race):
    PCA_df = [game for game in PCA_df if game != None]
    if race == None:
        PCA_df_ = [df_list[player][0] for df_list in PCA_df for player in range(0,len(df_list))]
    else:
        PCA_df_ = [df_list[player][0] for df_list in PCA_df for player in range(0,len(df_list)) if df_list[player][3] == race]
    PCA_df_o = pd.concat(PCA_df_, sort = False).fillna(0).drop(columns = ['game_id', 'player_id', 'second'])
    mm = MinMaxScaler()
    df_mm = mm.fit_transform(PCA_df_o)
    pca = PCA(n_components = 4)
    tr = pca.fit_transform(df_mm)
    explore_r4(tr[:,0], tr[:,1], tr[:,2], tr[:,3])

def pipeline(func = [None, PCA_UBE_df, PCA_operation], parameters =[None, 'UBE']):
    if parameters[0] == None:
        games = query().all()
    else:
        games = func[0](parameters[0])
    A = [func[1](game) for game in games]
    B = func[2](A, parameters[1])
    return B


if False:
    B_0, A_col_0 = pipeline(func = [filter_by_Game_highest_league, PCA_UBE_df, PCA_operation], parameters = [20, 'UBE'])
    B_1, A_col_1 = pipeline(func = [filter_by_Game_highest_league, PCA_PSE_df, PCA_operation], parameters = [20, 'PSE'])
    B_2, A_col_2 = pipeline(func = [filter_by_Game_highest_league, PCA_TPE_df, PCA_operation], parameters = [20, 'TPE'])
    B_3, A_col_3 = pipeline(func = [filter_by_Game_highest_league, PCA_UDE_df, PCA_operation], parameters = [20, 'UDE'])
    B_4, A_col_4 = pipeline(func = [filter_by_Game_highest_league, PCA_BCE_df, PCA_operation], parameters = [20, 'BCE'])
print('exit PCA')
