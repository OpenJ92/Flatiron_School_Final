#### Extract UnitBornEvents / UnitDonEvents by game.highest_league
#### and construct dataframe of aggregate actions and second.
#### Then carry out PCA and grab the first principle component.
#### This is a vector that describes our data. Then map to a hyper-sphere
#### of radius R (?) and carry out Kmeans on full set. This, I believe is a
#### reasonable measure of game stratagy.

### Goal for tonight. Go through professional Replays and carry out TruncatedSVD
### on each game. Use first principle component to carry out classificationself.
### Show that with 3 classes, one should find individual races.

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
import os
offline.init_notebook_mode()

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||Reconstruction under new schema
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def event_Dictionary():
    UBE_t = ['Banshee', 'Cyclone', 'Marine', 'Medivac', 'Raven', 'Reaper', 'SiegeTank', 'VikingFighter', 'Hellion', 'Liberator', 'Thor', 'Marauder', 'WidowMine', 'HellionTank', 'Battlecruiser', 'Ghost', 'SCV']
    UBE_z = ['Roach', 'Baneling',  'Mutalisk', 'Queen', 'Zergling', 'Corruptor', 'Hydralisk', 'Viper', 'Ultralisk', 'Drone']
    UBE_p = ['Stalker', 'Colossus', 'Disruptor', 'Immortal', 'WarpPrism', 'Observer', 'Adept', 'Phoenix', 'Oracle', 'Zealot', 'Sentry', 'Tempest', 'Carrier', 'VoidRay', 'Archon', 'Mothership', 'HighTemplar', 'Probe']

    PSE_coldrop = ['name', 'player', '_sa_instance_state']

    TPE_t = ['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab', 'BuildCommandCenter', 'BuildEngineeringBay', 'BuildFactory', 'BuildFactoryTechLab', 'BuildMissileTurret', 'BuildSensorTower', 'BuildStarport', 'BuildStarportReactor', 'BuildStarportTechLab', 'BuildSupplyDepot', 'BuildFactoryReactor', 'BuildBunker', 'BuildFusionCore', 'BuildGhostAcademy']
    TPE_z = ['BuildHatchery', 'BuildRoachWarren', 'BuildSpawningPool', 'BuildBanelingNest', 'BuildCreepTumor', 'BuildEvolutionChamber', 'BuildSpire', 'BuildSporeCrawler', 'BuildHydraliskDen', 'BuildInfestationPit', 'BuildSpineCrawler', 'BuildUltraliskCavern', 'BuildNydusNetwork', 'BuildLurkerDenMP']
    TPE_p = ['BuildCyberneticsCore', 'BuildDarkShrine', 'BuildForge', 'BuildGateway', 'BuildNexus', 'BuildPylon', 'BuildShieldBattery', 'BuildTemplarArchive', 'BuildTwilightCouncil', 'BuildRoboticsBay', 'BuildRoboticsFacility', 'BuildStargate', 'BuildPhotonCannon', 'BuildFleetBeacon']

    UDE_t = ['Armory', 'Barracks', 'BarracksReactor', 'EngineeringBay', 'Factory', 'FactoryTechLab', 'MissileTurret', 'OrbitalCommand', 'PlanetaryFortress', 'Refinery', 'SensorTower', 'Starport', 'StarportReactor', 'StarportTechLab', 'BarracksTechLab', 'CommandCenter', 'FactoryReactor', 'Bunker', 'FusionCore', 'GhostAcademy']
    UDE_z = ['Hatchery', 'RoachWarren', 'SpawningPool', 'BanelingNest', 'EvolutionChamber', 'Spire', 'GreaterSpire', 'HydraliskDen', 'InfestationPit', 'SpineCrawler', 'SporeCrawler', 'UltraliskCavern', 'Lair', 'NydusNetwork', 'Hive', 'CreepTumorQueen', 'CreepTumor', 'LurkerDen']
    UDE_p = ['CyberneticsCore', 'DarkShrine', 'Forge', 'Nexus', 'ShieldBattery', 'TemplarArchive', 'TwilightCouncil', 'WarpGate','RoboticsBay', 'RoboticsFacility', 'Stargate', 'PhotonCannon', 'FleetBeacon', 'Gateway']

    BCE_t = ['BuildSiegeTank', 'TrainBanshee', 'TrainCyclone', 'TrainLiberator', 'TrainMarine', 'TrainMedivac', 'TrainRaven', 'TrainReaper', 'TrainViking', 'BuildHellion', 'BuildThor', 'TrainMarauder', 'BuildWidowMine', 'TrainBattlecruiser', 'TrainGhost', 'BuildBattleHellion', 'TrainNuke', 'TrainSCV']
    BCE_z = ['MorphRoach', 'MorphToRavager', 'MorphMutalisk', 'MorphToOverseer', 'MorphZergling', 'TrainBaneling', 'TrainQueen', 'MorphCorruptor', 'MorphHydralisk', 'MorphInfestor', 'MorphSwarmHost', 'MorphToBroodLord', 'MorphViper', 'MorphUltralisk', 'TrainDrone']
    BCE_p = ['TrainStalker', 'TrainColossus', 'TrainDisruptor', 'TrainImmortal', 'TrainObserver', 'TrainWarpPrism', 'TrainAdept', 'TrainZealot', 'TrainOracle', 'TrainPhoenix', 'TrainSentry', 'TrainTempest', 'TrainCarrier', 'TrainVoidRay', 'TrainMothership', 'TrainInterceptor', 'TrainProbe']

    return {'UBE': {'Terran': UBE_t,'Zerg': UBE_z,'Protoss': UBE_p, 'event_column': 'unit_type_name'},
            'TPE': {'Terran': TPE_t,'Zerg': TPE_z,'Protoss': TPE_p, 'event_column': 'ability_name'},
            'UDE': {'Terran': UDE_t,'Zerg': UDE_z,'Protoss': UDE_p, 'event_column': 'unit'},
            'BCE': {'Terran': BCE_t,'Zerg': BCE_z,'Protoss': BCE_p, 'event_column': 'ability_name'},
            'drop_add': ['participant_id', 'second']}

def unique_event_names():
    event_dictionary_ = event_Dictionary()
    return {'UBE': event_dictionary_['UBE']['Terran'] + event_dictionary_['UBE']['Zerg'] + event_dictionary_['UBE']['Protoss'],
            'TPE': event_dictionary_['TPE']['Terran'] + event_dictionary_['TPE']['Zerg'] + event_dictionary_['TPE']['Protoss'],
            'UDE': event_dictionary_['UDE']['Terran'] + event_dictionary_['UDE']['Zerg'] + event_dictionary_['UDE']['Protoss'],
            'BCE': event_dictionary_['BCE']['Terran'] + event_dictionary_['BCE']['Zerg'] + event_dictionary_['BCE']['Protoss']}

def explore_r4(x,y,z,u,name):
    traces = go.Scatter3d(x = np.array(x), y = np.array(y), z = np.array(z), mode='markers', marker=dict(size = 3, color = u, colorscale = 'Jet', opacity = 0))
    #Label axes etc etc....
    fig = go.Figure(data=[traces])
    offline.plot(fig, filename = 'plotly_files/' + name + '.html', auto_open=False)
    return None

def aggregate_cumulative_events(df_drop_add, df_dummy):
    df_dummy = df_dummy.cumsum()
    return pd.concat([df_dummy, df_drop_add], axis = 1, sort = False)

def aggregate_cumulative_time_events(aggregate_cumulative_events):
    min_ = aggregate_cumulative_events['second'].min()
    max_ = aggregate_cumulative_events['second'].max()
    second_ = list(range(int(min_), int(max_)))
    second_df = pd.DataFrame(second_, columns = ['second'])
    ACTE = pd.merge(aggregate_cumulative_events, second_df, how = 'right', on = 'second').sort_values(by = ['second']).ffill()
    return ACTE[~ACTE['second'].duplicated(keep='last')]

def construct_df_UnitsStructures(participant, event_name, time = False):
    participant_events = participant.events_(event_name)
    participant_game_id = participant.game[0].id
    event_Dictionary_ = event_Dictionary()

    try:
        df_event = pd.DataFrame([vars(event) for event in participant_events]).sort_values(by = ['second'])
    except:
        return None

    df_event_gd = pd.get_dummies(df_event[event_Dictionary_[event_name]['event_column']])
    df_event_gd_filter = df_event_gd[[col for col in df_event_gd.columns if col in event_Dictionary_[event_name][participant.playrace]]]
    df_event_gd_agg = aggregate_cumulative_events(df_event[event_Dictionary_['drop_add']], df_event_gd_filter)

    if time:
        df_event_gd_agg = aggregate_cumulative_time_events(df_event_gd_agg)

    df_event_gd_agg = df_event_gd_agg[~df_event_gd_agg['second'].duplicated(keep='last')]
    return df_event_gd_agg

def construct_full_UnitsStructures_df(participant, event_name, time = False):
    participant_df_UnitsStructures = construct_df_UnitsStructures(participant, event_name, time = time)
    full_col = unique_event_names()[event_name]
    full_DataFrame = pd.DataFrame(columns = full_col)
    return pd.concat([participant_df_UnitsStructures, full_DataFrame], axis = 0, sort = False).fillna(0)[full_col + ['second', 'participant_id']]

def combine_df_UnitStructures(participant, list_event_name):
    _df_ = [construct_df_UnitsStructures(participant, event, time = True) for event in list_event_name]
    f = lambda x,y: pd.merge(x,y,right_on = 'second')
    #ittertools reduce to merge these functions. return as a DataFrame

def combine_full_df_UnitStructures(participant, list_event_name):
    _df_ = [construct_full_UnitsStructures_df(participant, event, time = True) for event in list_event_name]
    f = lambda x,y: pd.merge(x,y,right_on = 'second')
    #ittertools reduce to merge these functions. return as a DataFrame

def fit_construct_PCAoTSVD(participant, event_name, time = False, func_decomp = PCA, func_normalization = MinMaxScaler, name_decomp = 'PCA', name_normalization = 'MinMax'):
    construct_full_UnitsStructures_df_ = construct_full_UnitsStructures_df(participant, event_name, time).drop(columns = ['second', 'participant_id'])
    decomposition_analysis = func_decomp()
    normalization_scalar = func_normalization()
    construct_full_UnitsStructures_df_mm = normalization_scalar.fit_transform(construct_full_UnitsStructures_df_)
    decomposition_analysis.fit(construct_full_UnitsStructures_df_mm)
    None if os.path.exists(name_decomp + '_' + name_normalization + '_Models/') else os.mkdir(name_decomp + '_' + name_normalization + '_Models/')
    joblib.dump(decomposition_analysis,
                name_decomp + '_' + name_normalization + '_Models/' + event_name + '_' +
                str(participant.id) + '_' + str(participant.user[0].id) + '_' + participant.playrace + '_' +
                str(participant.game[0].id) + '_' + participant.league +'.joblib')

def pipeline(event_name, time = True, func_decomp = PCA, func_normalization = MinMaxScaler, name_decomp = 'PCA', name_normalization = 'MinMax'):
    participants = db.session.query(Participant)[:100]
    [fit_construct_PCAoTSVD(participant, event_name, time, func_decomp, func_normalization, name_decomp, name_normalization) for participant in participants]

# ?Move to unsupervised.py

def load_Decomposition(participant, name_decomp, name_normalization, event_name):
    decomposition_load = joblib.load(name_decomp + '_' + name_normalization + '_Models/' + event_name + '_' +
    str(participant.id) + '_' + str(participant.user[0].id) + '_' + participant.playrace + '_' +
    str(participant.game[0].id) + '_' + participant.league +'.joblib')
    return decomposition_load

def load_Decomposition_FSV(participant, name_decomp, name_normalization, event_name):
    return load_Decomposition(participant, name_decomp, name_normalization, event_name).components_[0]

def load_Decomposiation_batch(sql_func, name_decomp, name_normalization, event_name):
    participants = sql_func()
    return [load_Decomposition(participant, name_decomp, name_normalization, event_name) for participant in participants]

def load_Decomposition_batch_FSV(sql_func, name_decomp, name_normalization, event_name):
    participants = sql_func()
    singular_vector_decomposition = [load_Decomposition_FSV(participant, name_decomp, name_normalization, event_name)for participant in participants]
    singular_vector_decomposition_DataFrame = pd.concat(singular_vector_decomposition, axis = 0, columns = unique_event_names()[event_name], sort = False).T
    return singular_vector_decomposition_DataFrame

def plot_(DataFrame, name):
    #For use with load_Decomposition_batch_FSV, construct/combine_full_UnitsStructures_df, construct/combine_df_UnitsStructures
    principle_component_analysis = PCA(n_components = 4)
    X = principle_component_analysis.fit_transform(DataFrame)
    explore_r4(X[:0], X[:1], X[:2], X[:3], name)

def radial_RSS(participant):
    singular_vector = load_Decomposition_FSV(participant, name_decomp, name_normalization, event_name)
    event_df = construct_full_UnitsStructures_df(participant, event_name, time = False)
    sum((event_df @ len(event_df[i])singular_vector) / len(event_df[i])**2) ??

print('exit PCA')
