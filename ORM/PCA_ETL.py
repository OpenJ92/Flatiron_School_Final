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
from functools import reduce
from itertools import combinations
import os
offline.init_notebook_mode()

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||Reconstruction under new schema
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def event_Dictionary():
    UBE_t = ['Banshee', 'Cyclone', 'Marine', 'Medivac', 'Raven', 'Reaper', 'SiegeTank', 'VikingFighter', 'Hellion', 'Liberator', 'Thor', 'Marauder', 'WidowMine', 'HellionTank', 'Battlecruiser', 'Ghost', 'SCV']
    UBE_z = ['Roach', 'Baneling',  'Mutalisk', 'Queen', 'Zergling', 'Corruptor', 'Hydralisk', 'Viper', 'Ultralisk', 'Drone', 'Overlord']
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

def explore_r3(x,y,z,name):
    traces = go.Scatter3d(x = np.array(x), y = np.array(y), z = np.array(z), mode='markers', marker = dict(size=3))
    #Label axes etc etc....
    fig = go.Figure(data=[traces])
    offline.plot(fig, filename = 'plotly_files/' + name + '.html', auto_open=True)
    return None

def aggregate_cumulative_events(df_drop_add, df_dummy):
    df_dummy = df_dummy.cumsum()
    #import pdb; pdb.set_trace()
    return pd.concat([df_dummy, df_drop_add], axis = 1, sort = False)

def aggregate_cumulative_time_events(aggregate_cumulative_events):
    min_ = aggregate_cumulative_events['second'].min()
    max_ = aggregate_cumulative_events['second'].max()
    second_ = list(range(int(min_), int(max_)))
    second_df = pd.DataFrame(second_, columns = ['second'])
    ACTE = pd.merge(aggregate_cumulative_events, second_df, how = 'right', on = 'second').sort_values(by = ['second']).ffill()
    #import pdb; pdb.set_trace()
    return ACTE[~ACTE['second'].duplicated(keep='last')]

def construct_df_UnitsStructures(participant, event_name, aggregate = True, time = False):
    participant_events = participant.events_(event_name)
    participant_game_id = participant.game[0].id
    event_Dictionary_ = event_Dictionary()

    try:
        df_event = pd.DataFrame([vars(event) for event in participant_events]).sort_values(by = ['second'])
    except:
        return None

    #import pdb; pdb.set_trace()

    df_event_gd = pd.get_dummies(df_event[event_Dictionary_[event_name]['event_column']])
    df_event_gd_filter = df_event_gd[[col for col in df_event_gd.columns if col in unique_event_names()[event_name]]]
    df_event_gd_agg = df_event_gd_filter

    if aggregate:
        #import pdb; pdb.set_trace()
        df_event_gd_agg = aggregate_cumulative_events(df_event[event_Dictionary_['drop_add']], df_event_gd_filter)

    if time:
        #import pdb; pdb.set_trace()
        if not aggregate:
            #import pdb; pdb.set_trace()
            df_event_gd_agg = pd.concat([df_event_gd_filter, df_event[event_Dictionary_['drop_add']]], axis = 1, sort = False)
        df_event_gd_agg = aggregate_cumulative_time_events(df_event_gd_agg)

    if not time and not aggregate:
        df_event_gd_agg = pd.concat([df_event_gd_filter, df_event[['second', 'participant_id']]], axis = 1, sort = False)

    df_event_gd_agg = df_event_gd_agg[~df_event_gd_agg['second'].duplicated(keep='last')]
    return df_event_gd_agg

def construct_full_UnitsStructures_df(participant, event_name, aggregate = True, time = False):
    participant_df_UnitsStructures = construct_df_UnitsStructures(participant, event_name, aggregate, time)
    full_col = unique_event_names()[event_name]
    full_DataFrame = pd.DataFrame(columns = full_col)
    #import pdb; pdb.set_trace()
    return pd.concat([participant_df_UnitsStructures, full_DataFrame], axis = 0, sort = False).fillna(0)[full_col + ['second', 'participant_id']]

def combine_df_UnitStructures(participant, list_event_name, aggragate_):
    _df_ = [construct_full_UnitsStructures_df(participant, event, aggragate_, True) for event in list_event_name]
    _df__ = reduce(lambda x,y: pd.merge(right = x, left = y, on = 'second', how = 'outer').fillna(0), _df_)
    _df__ = _df__[reduce(lambda x,y: x+y,[unique_event_names()[event_name] for event_name in sorted(list_event_name)])]
    return _df__

# _____________________________ up to this point, all functions are functioning.

def fit_construct_PCAoTSVD(participant, event_name, time = False, aggregate = True, func_decomp = PCA, func_normalization = MinMaxScaler, name_decomp = 'PCA', name_normalization = 'MinMax'):
    _path = event_name + '_' + name_decomp + '_' + name_normalization + '_time' + str(time) + '_agg' + str(aggregate) + '_Models/'
    path_ = event_name + '_' + str(participant.id) + '_' + str(participant.user[0].id) + '_' + participant.playrace + '_' + str(participant.game[0].id) + '_' + participant.league +'.joblib'
    if os.path.exists(_path + path_):
        print('File already exists: ' + _path + path_)
        return None
    construct_full_UnitsStructures_df_ = construct_full_UnitsStructures_df(participant, event_name, time, aggregate).drop(columns = ['second', 'participant_id'])
    decomposition_analysis = func_decomp()
    normalization_scalar = func_normalization()
    try:
        construct_full_UnitsStructures_df_mm = normalization_scalar.fit_transform(construct_full_UnitsStructures_df_)
        decomposition_analysis.fit(construct_full_UnitsStructures_df_mm)
        None if os.path.exists(_path) else os.mkdir(_path)
        joblib.dump(decomposition_analysis, _path + path_)
    except Exception as e:
        print(e)

def fit_construct_PCAoTSVD_combined(participant, event_names, aggregate = True, func_decomp = PCA, func_normalization = MinMaxScaler, name_decomp = 'PCA', name_normalization = 'MinMax'):
    _path = str(sorted(event_names)) + '_' +name_decomp + '_' + name_normalization + '_time' + 'True' + '_agg' + str(aggregate) + '_c_Models/'
    path_ = str(sorted(event_names)) + '_' + str(participant.id) + '_' + str(participant.user[0].id) + '_' + participant.playrace + '_' + str(participant.game[0].id) + '_' + participant.league +'.joblib'
    if os.path.exists(_path + path_):
        print('File already exists: ' + _path + path_)
        return None
    #import pdb; pdb.set_trace()
    combine_df_UnitStructures_ = combine_df_UnitStructures(participant, event_names, aggregate)
    combine_df_UnitStructures_ = combine_df_UnitStructures_.drop(columns = [col for col in combine_df_UnitStructures_.columns if col in 'participant_id'])
    decomposition_analysis = func_decomp()
    normalization_scalar = func_normalization()
    try:
        combine_df_UnitStructures_mm = normalization_scalar.fit_transform(combine_df_UnitStructures_)
        decomposition_analysis.fit(combine_df_UnitStructures_mm)
        None if os.path.exists(_path) else os.mkdir(_path)
        joblib.dump(decomposition_analysis, _path + path_)
    except Exception as e:
        print(e)
        pass

def pipeline(sql_func, func_decomp = PCA, func_normalization = MinMaxScaler, name_decomp = 'PCA', name_normalization = 'MinMax'):
    participants = sql_func()
    event_list = ['UDE', 'BCE', 'TPE', 'UBE']
    agg_time = [True, False]

    for event_name in event_list:
        for a_t in list(combinations(agg_time, 2)):
            for participant in participants:
                try:
                    fit_construct_PCAoTSVD(participant, event_name, a_t[0], a_t[1], func_decomp, func_normalization, name_decomp, name_normalization)
                except Exception as e:
                    print(e)

def combined_pipeline(sql_func, func_decomp = PCA, func_normalization = MinMaxScaler, name_decomp = 'PCA', name_normalization = 'MinMax'):
    participants = sql_func()
    event_list = ['UDE', 'BCE', 'TPE', 'UBE']
    agg_time = [True, False]

    for i in range(2,len(event_list)):
        for event_combs in list(combinations(event_list, i)):
            for a_t in agg_time:
                for participant in participants:
                    try:
                        fit_construct_PCAoTSVD_combined(participant, event_combs, a_t)
                    except Exception as e:
                        print(e)

combined_pipeline(db.session.query(Participant).filter(Participant.league == 20).all)
pipeline(db.session.query(Participant).filter(Participant.league == 20).all)

## Completed Pipeline queries:
##      1. db.session.query(Participant).filter(Participant.league == 20).all (in_Progress)

print('exit PCA')
