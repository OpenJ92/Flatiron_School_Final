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
            'drop_add': ['game_id', 'player_id', 'second']}

def aggregate_cumulative_events(df_drop_add, df_dummy):
    df_dummy = df_dummy.cumsum()
    return pd.concat([df_dummy, df_drop_add], axis = 1, sort = False)

def aggregate_cumulative_time_events(aggregate_cumulative_events):
    min_ = aggregate_cumulative_events['second'].min()
    max_ = aggregate_cumulative_events['second'].max()
    second_ = list(range(min_, max_))
    second_df = pd.DataFrame(second_, columns = ['second'])
    #You're working on this currently.

def _df_UnitsStructures(participant, event_name, event_method, time = False):
    participant_events = participant.event_method
    participant_game_id = participant.game.id
    event_Dictionary = event_Dictionary()

    try:
        df_event = pd.DataFrame([vars(event) for event in participant_events]).sort_values(by = ['second'])
    except:
        return None

    df_event_gd = pd.get_dummies(df_event[event_Dictionary[event_name]['event_column']])[event_Dictionary[event_name][participant.playrace]]
    df_event_gd_agg = aggregate_cumulative_events(df_event[event_Dictionary['drop_add']], df_event_gd)

    if time:
        df_event_gd_agg = aggregate_cumulative_time_events(df_event_gd_agg)

    return df_event_gd_agg, participant_game_id, participant.id

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||DataFrame Construction||Feature Selection
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    def explore_r4(x,y,z,u,name):
         traces = go.Scatter3d(x = np.array(x), y = np.array(y), z = np.array(z), mode='markers', marker=dict(size = 3, color = u, colorscale = 'Jet', opacity = 0))
         fig = go.Figure(data=[traces])
         offline.plot(fig, filename = 'plotly_files/' + name + '.html', auto_open=False)
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
            df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Banshee', 'Cyclone', 'Marine', 'Medivac', 'Raven', 'Reaper',
                                                                                            'SiegeTank', 'VikingFighter', 'Hellion', 'Liberator', 'Thor', 'Marauder',
                                                                                            'WidowMine', 'HellionTank', 'Battlecruiser', 'Ghost', 'SCV']]]
        elif game.playerOne_playrace == 'Protoss':
            df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Stalker', 'Colossus', 'Disruptor', 'Immortal', 'WarpPrism', 'Observer',
                                                                                            'Adept', 'Phoenix', 'Oracle', 'Zealot', 'Sentry', 'Tempest', 'Carrier',
                                                                                            'VoidRay', 'Archon', 'Mothership', 'HighTemplar', 'Probe']]]
        else:
            df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['Roach', 'Baneling',  'Mutalisk', 'Queen', 'Zergling', 'Corruptor', 'Hydralisk',
                                                                                            'Viper', 'Ultralisk', 'Drone']]]
        if game.playerTwo_playrace == 'Terran':
            df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Banshee', 'Cyclone', 'Marine', 'Medivac', 'Raven', 'Reaper',
                                                                                            'SiegeTank', 'VikingFighter', 'Hellion', 'Liberator', 'Thor', 'Marauder',
                                                                                            'WidowMine', 'HellionTank', 'Battlecruiser', 'Ghost', 'SCV']]]
        elif game.playerTwo_playrace == 'Protoss':
            df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Stalker', 'Colossus', 'Disruptor', 'Immortal', 'WarpPrism', 'Observer',
                                                                                            'Adept', 'Phoenix', 'Oracle', 'Zealot', 'Sentry', 'Tempest', 'Carrier',
                                                                                            'VoidRay', 'Archon', 'Mothership', 'HighTemplar', 'Probe']]]
        else:
            df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['Roach', 'Baneling',  'Mutalisk', 'Queen', 'Zergling', 'Corruptor', 'Hydralisk',
                                                                                            'Viper', 'Ultralisk', 'Drone']]]

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
                                            'TrainBattlecruiser', 'TrainGhost', 'BuildBattleHellion', 'TrainNuke', 'TrainSCV']]]
        elif game.playerOne_playrace == 'Protoss':
            df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['TrainStalker', 'TrainColossus', 'TrainDisruptor', 'TrainImmortal', 'TrainObserver', 'TrainWarpPrism', 'TrainAdept',
                                            'TrainZealot', 'TrainOracle', 'TrainPhoenix', 'TrainSentry', 'TrainTempest', 'TrainCarrier', 'TrainVoidRay', 'TrainMothership', 'TrainInterceptor', 'TrainProbe']]]
        else:
            df_PlayerOne_ = df_PlayerOne_[[col for col in df_PlayerOne_.columns if col in ['MorphRoach', 'MorphToRavager', 'MorphMutalisk', 'MorphToOverseer', 'MorphZergling', 'TrainBaneling', 'TrainQueen',
                                            'MorphCorruptor', 'MorphHydralisk', 'MorphInfestor', 'MorphSwarmHost', 'MorphToBroodLord', 'MorphViper', 'MorphUltralisk', 'TrainDrone']]]
        if game.playerTwo_playrace == 'Terran':
            df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['BuildSiegeTank', 'TrainBanshee', 'TrainCyclone', 'TrainLiberator', 'TrainMarine', 'TrainMedivac',
                                            'TrainRaven', 'TrainReaper', 'TrainViking', 'BuildHellion', 'BuildThor', 'TrainMarauder', 'BuildWidowMine',
                                            'TrainBattlecruiser', 'TrainGhost', 'BuildBattleHellion', 'TrainNuke', 'TrainSCV']]]
        elif game.playerTwo_playrace == 'Protoss':
            df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['TrainStalker', 'TrainColossus', 'TrainDisruptor', 'TrainImmortal', 'TrainObserver', 'TrainWarpPrism', 'TrainAdept',
                                            'TrainZealot', 'TrainOracle', 'TrainPhoenix', 'TrainSentry', 'TrainTempest', 'TrainCarrier', 'TrainVoidRay', 'TrainMothership', 'TrainInterceptor', 'TrainProbe']]]
        else:
            df_PlayerTwo_ = df_PlayerTwo_[[col for col in df_PlayerTwo_.columns if col in ['MorphRoach', 'MorphToRavager', 'MorphMutalisk', 'MorphToOverseer', 'MorphZergling', 'TrainBaneling', 'TrainQueen',
                                            'MorphCorruptor', 'MorphHydralisk', 'MorphInfestor', 'MorphSwarmHost', 'MorphToBroodLord', 'MorphViper', 'MorphUltralisk', 'TrainDrone']]]

        #import pdb; pdb.set_trace()

        df_PlayerOne_o = pd.concat([df_PlayerOne_, df_PlayerOne[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
        df_PlayerTwo_o = pd.concat([df_PlayerTwo_, df_PlayerTwo[['game_id', 'player_id', 'second']]], axis = 1, sort = False)
        #expected output -> [[df, player_id, game_id], [df, player_id, game_id]]
        return [df_PlayerOne_o, game_players[0].id, game_id, game.playerOne_playrace], [df_PlayerTwo_o, game_players[1].id, game_id, game.playerTwo_playrace]

    def PCA_operation(events_PCA, event_, list_of_games):
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

        for game in list_of_games:
            for player in game.players:
                #import pdb; pdb.set_trace()
                events_PCA_conglomerate_temp = events_PCA_conglomerate_df[(events_PCA_conglomerate_df['player_id'] == player.id) & (events_PCA_conglomerate_df['game_id'] == game.id)]
                events_PCA_conglomerate_temp_ = events_PCA_conglomerate_temp.drop(columns = ['player_id', 'game_id', 'second'])
                if (events_PCA_conglomerate_temp_.shape[1] == 0) or (events_PCA_conglomerate_temp_.shape[0] == 0):
                    break
                mm = MinMaxScaler()
                df_mm = mm.fit_transform(events_PCA_conglomerate_temp_)
                _PCA = PCA()
                _PCA.fit(df_mm)
                joblib.dump(_PCA, 'PCA_Models_'+ event_ +'/' + str(player.id) + '_' + str(game.id) + '.joblib')

        # for game in range(0,len(events_PCA)):
        #     for player in range(0,len(events_PCA[game])):
        #         df = events_PCA[game][player][0].drop(columns = ['game_id', 'player_id', 'second'])
        #         if df.shape[1] == 0:
        #             break
        #         mm = MinMaxScaler()
        #         df_mm = mm.fit_transform(df)
        #         _PCA = PCA()
        #         _PCA.fit(df_mm)
        #         ## 'player_id'_'game_id'.joblib
        #         joblib.dump(_PCA, 'PCA_Models_'+ event_ +'/' + str(events_PCA[game][player][1]) + '_' + str(events_PCA[game][player][2]) + '.joblib')
        #         events_PCA[game][player].append(_PCA)

        #expected output -> [[[df, player_id, game_id, df_A, PCAobject], [df, player_id, game_id, df_A, PCAobject]], [[df, player_id, game_id, df_A, PCAobject], ..., [df, player_id, game_id, df_A, PCAobject]]
        return events_PCA_conglomerate_df #events_PCA, events_PCA[0][0][0].columns

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

    def plot_df_vectors(PCA_df, race, name):
        PCA_df = [game for game in PCA_df if game != None]
        if race == None:
            PCA_df_ = [df_list[player][0] for df_list in PCA_df for player in range(0,len(df_list))]
        else:
            PCA_df_ = [df_list[player][0] for df_list in PCA_df for player in range(0,len(df_list)) if df_list[player][3] == race]
        PCA_df_o = pd.concat(PCA_df_, sort = False).fillna(0).drop(columns = ['game_id', 'player_id', 'second'])
        mm = MinMaxScaler()
        df_mm = mm.fit_transform(PCA_df_o)
        pca = PCA()
        tr = pca.fit_transform(df_mm)
        explore_r4(tr[:,0], tr[:,1], tr[:,2], tr[:,3], name)

    def pipeline(func = [None, PCA_UBE_df, PCA_operation], parameters =[None, 'UBE']):
        if parameters[0] == None:
            games = query().all()
        else:
            games = func[0](parameters[0])
        A = [func[1](game) for game in games]
        B = func[2](A, parameters[1], parameters[2])
        return B

    #A = filter_by_Game_highest_league(20)

    if False:
        B_0, A_col_0 = pipeline(func = [filter_by_Game_highest_league, PCA_UBE_df, PCA_operation], parameters = [20, 'UBE', A]) #complete
        B_1, A_col_1 = pipeline(func = [filter_by_Game_highest_league, PCA_PSE_df, PCA_operation], parameters = [20, 'PSE', A]) #complete
        B_2, A_col_2 = pipeline(func = [filter_by_Game_highest_league, PCA_TPE_df, PCA_operation], parameters = [20, 'TPE', A]) #complete
        B_3, A_col_3 = pipeline(func = [filter_by_Game_highest_league, PCA_UDE_df, PCA_operation], parameters = [20, 'UDE', A]) #complete
        B_4, A_col_4 = pipeline(func = [filter_by_Game_highest_league, PCA_BCE_df, PCA_operation], parameters = [20, 'BCE', A]) #complete

print('exit PCA')
