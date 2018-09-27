import sc2reader
import bs4
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import webbrowser
from functools import reduce
import numpy as np
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()



if False: #Get replay files from ST.
    _link_PvT = 'https://lotv.spawningtool.com/replays/?pro_only=on&query=&after_played_on=&before_played_on=&after_time=&order_by=&before_time=&tag=10&tag=1&coop=n&patch=130&p='
    ST_PvT_Requests = [requests.get(_link_PvT + str(link_)) for link_ in range(1, 22)]
    ST_PvT_BeautifulSoup = [BeautifulSoup(request.text) for request in ST_PvT_Requests]
    ST_PvT_A = [Beautiful.find_all('a') for Beautiful in ST_PvT_BeautifulSoup]
    ST_PvT_DL = ['https://lotv.spawningtool.com' + element['href'] for Beautiful in ST_PvT_A for element in Beautiful if 'download' in element['href']]
    ST_PvT_file_to_dir = [webbrowser.open(link) for link in ST_PvT_DL]

    _link_TvT = 'https://lotv.spawningtool.com/replays/?pro_only=on&query=&after_played_on=&before_played_on=&after_time=&order_by=&before_time=&tag=12&tag=1&coop=n&patch=130&p='
    ST_TvT_Requests = [requests.get(_link_TvT + str(link_)) for link_ in range(1, 8)]
    ST_TvT_BeautifulSoup = [BeautifulSoup(request.text) for request in ST_TvT_Requests]
    ST_TvT_A = [Beautiful.find_all('a') for Beautiful in ST_TvT_BeautifulSoup]
    ST_TvT_DL = ['https://lotv.spawningtool.com' + element['href'] for Beautiful in ST_TvT_A for element in Beautiful if 'download' in element['href']]
    ST_TvT_file_to_dir = [webbrowser.open(link) for link in ST_TvT_DL]

    _link_ZvT = 'https://lotv.spawningtool.com/replays/?pro_only=on&coop=n&before_time=&after_time=&order_by=&before_played_on=&query=&tag=3&tag=1&after_played_on=&patch=130&p='
    ST_ZvT_Requests = [requests.get(_link_ZvT + str(link_)) for link_ in range(1, 27)]
    ST_ZvT_BeautifulSoup = [BeautifulSoup(request.text) for request in ST_ZvT_Requests]
    ST_ZvT_A = [Beautiful.find_all('a') for Beautiful in ST_ZvT_BeautifulSoup]
    ST_ZvT_DL = ['https://lotv.spawningtool.com' + element['href'] for Beautiful in ST_ZvT_A for element in Beautiful if 'download' in element['href']]
    ST_ZvT_file_to_dir = [webbrowser.open(link) for link in ST_ZvT_DL]

    _link_ZvZ = 'https://lotv.spawningtool.com/replays/?tag=13&tag=2&before_time=&after_time=&after_played_on=&coop=n&query=&patch=130&order_by=&before_played_on=&pro_only=on&p='
    ST_ZvZ_Requests = [requests.get(_link_ZvZ + str(link_)) for link_ in range(1, 23)]
    ST_ZvZ_BeautifulSoup = [BeautifulSoup(request.text) for request in ST_ZvZ_Requests]
    ST_ZvZ_A = [Beautiful.find_all('a') for Beautiful in ST_ZvZ_BeautifulSoup]
    ST_ZvZ_DL = ['https://lotv.spawningtool.com' + element['href'] for Beautiful in ST_ZvZ_A for element in Beautiful if 'download' in element['href']]
    ST_ZvZ_file_to_dir = [webbrowser.open(link) for link in ST_ZvZ_DL]

    _link_ZvP = 'https://lotv.spawningtool.com/replays/?before_time=&tag=11&tag=2&order_by=&before_played_on=&pro_only=on&patch=130&after_time=&query=&after_played_on=&coop=n&p='
    ST_ZvP_Requests = [requests.get(_link_ZvP + str(link_)) for link_ in range(1, 33)]
    ST_ZvP_BeautifulSoup = [BeautifulSoup(request.text) for request in ST_ZvP_Requests]
    ST_ZvP_A = [Beautiful.find_all('a') for Beautiful in ST_ZvP_BeautifulSoup]
    ST_ZvP_DL = ['https://lotv.spawningtool.com' + element['href'] for Beautiful in ST_ZvP_A for element in Beautiful if 'download' in element['href']]
    ST_ZvP_file_to_dir = [webbrowser.open(link) for link in ST_ZvP_DL]

_link_PvP = 'https://lotv.spawningtool.com/replays/?tag=9&tag=17&before_time=&after_time=&after_played_on=&coop=n&query=&patch=130&order_by=&before_played_on=&pro_only=on&p='
ST_PvP_Requests = [requests.get(_link_PvP + str(link_)) for link_ in range(1, 11)]
ST_PvP_BeautifulSoup = [BeautifulSoup(request.text) for request in ST_PvP_Requests]
ST_PvP_A = [Beautiful.find_all('a') for Beautiful in ST_PvP_BeautifulSoup]
ST_PvP_DL = ['https://lotv.spawningtool.com' + element['href'] for Beautiful in ST_PvP_A for element in Beautiful if 'download' in element['href']]
ST_PvP_file_to_dir = [webbrowser.open(link) for link in ST_PvP_DL]

if False:
    replays_PvT = []
    replays_TvT = []
    replays_ZvT = []

    for replay in os.listdir('/Users/flatironschool/Desktop/PySC2/sc2reader/sc2_Replays/STReplaysPvT')[:1]:
        try:
            replays_PvT.append(sc2reader.load_replay('./sc2_Replays/STReplaysPvT/' + replay))
            print('./sc2_Replays/STReplaysPvT/' + replay)
        except:
            print('./sc2_Replays/STReplaysPvT/' + replay + '|||||||||||||||||')

    for replay in os.listdir('/Users/flatironschool/Desktop/PySC2/sc2reader/sc2_Replays/STReplaysTvT')[:1]:
        try:
            replays_TvT.append(sc2reader.load_replay('./sc2_Replays/STReplaysTvT/' + replay))
            print('./sc2_Replays/STReplaysTvT/' + replay)
        except:
            print('./sc2_Replays/STReplaysTvT/' + replay + '|||||||||||||||||')

    for replay in os.listdir('/Users/flatironschool/Desktop/PySC2/sc2reader/sc2_Replays/STReplaysZvT'):
        try:
            replays_ZvT.append(sc2reader.load_replay('./sc2_Replays/STReplaysZvT/' + replay))
            print('./sc2_Replays/STReplaysZvT/' + replay)
        except:
            print('./sc2_Replays/STReplaysZvT/' + replay + '|||||||||||||||||')

    version_ = {'PvT': replays_PvT, 'TvT': replays_TvT, 'ZvT': replays_ZvT}
    _version = 'ZvT'

    replay_events_name = [event.name for event in version_[_version][0].events]
    replay_events_name_unique = list(set(replay_events_name))

    def explore_r5(x,y,z,w,u):
         traces = go.Scatter3d(x = np.array(x), y = np.array(y), z = np.array(z), mode='markers', marker=dict(size = 5*u, color = w, colorscale = 'Jet', opacity = 0))
         fig = go.Figure(data=[traces])
         offline.plot(fig)
         return None

    def explore_r4(x,y,z,w):
         traces = go.Scatter3d(x = np.array(x), y = np.array(y), z = np.array(z), mode='markers', marker=dict(size = 5, color = w, colorscale = 'Jet', opacity = 0))
         fig = go.Figure(data=[traces])
         offline.plot(fig)
         return None

    def replay_game_events_filter_by_name(name, replay_event_method):
        return [event for event in replay_event_method if event.name == name]

    def extract_replay_events(replays):
        replays_events = []
        for _replay in replays:
            replays_events.append((_replay, {name:replay_game_events_filter_by_name(name, _replay.events) for name in replay_events_name_unique}))
        return replays_events

    def ETL(_replays_events):
        replays_PlayerStatsEvent = []
        replays_BasicCommandEvent = []
        replays_TargetPointCommandEvent = []
        replays_UnitBornEvent = []
        replays_UnitDoneEvent = []

        for match in _replays_events:
            try:
            #|||||||BasicCommandEvent
                _BCE = []
                for event in match[1]['BasicCommandEvent']:
                    _BCE.append(vars(event))
                replays_BasicCommandEvent.append(pd.DataFrame(_BCE))

            #|||||||TargetPointCommandEvent
                _TPE = []
                for event in match[1]['TargetPointCommandEvent']:
                    _TPE.append(vars(event))
                replays_TargetPointCommandEvent.append(pd.DataFrame(_TPE))

            #|||||||PlayerStatsEvent
                _PSE = []
                _PSE_0 = []
                _PSE_1 = []
                for event in match[1]['PlayerStatsEvent']:
                    if event.player == match[0].player[1]:
                        _PSE_0.append(vars(event))
                    elif event.player == match[0].player[2]:
                        _PSE_1.append(vars(event))

                _PSE_P0_ = pd.DataFrame(_PSE_0)
                _PSE_P1_ = pd.DataFrame(_PSE_1)

                print('_PSE | ', match[0].player[1],' | ',match[0].winner,' | ',match[0].map_name)
                #_PSE_P0_ = _PSE_P0_[~_PSE_P0_['second'].duplicated(keep='last')]
                _PSE_P0_S_m = max(_PSE_P0_['second'])
                _PSE_P0_S_ = pd.DataFrame(list(range(0,_PSE_P0_S_m)), columns = ['second'])
                _PSE_P0_merge = pd.merge(_PSE_P0_, _PSE_P0_S_, on = 'second', how = 'right')
                _PSE_P0_df = _PSE_P0_merge.sort_values(by = 'second')
                _PSE_P0_df.index = _PSE_P0_df['second']
                _PSE_P0_df_i = _PSE_P0_df.interpolate(method = 'linear').ffill()

                print('_PSE | ', match[0].player[2],' | ',match[0].winner,' | ',match[0].map_name)
                #_PSE_P1_ = _PSE_P1_[~_PSE_P1_['second'].duplicated(keep='last')]
                _PSE_P1_S_m = max(_PSE_P1_['second'])
                _PSE_P1_S_ = pd.DataFrame(list(range(0,_PSE_P1_S_m)), columns = ['second'])
                _PSE_P1_merge = pd.merge(_PSE_P1_, _PSE_P1_S_, on = 'second', how = 'right')
                _PSE_P1_df = _PSE_P1_merge.sort_values(by = 'second')
                _PSE_P1_df.index = _PSE_P1_df['second']
                _PSE_P1_df_i = _PSE_P1_df.interpolate(method = 'linear').ffill()

                replays_PlayerStatsEvent.append(pd.concat([_PSE_P0_df_i, _PSE_P1_df_i], axis = 0, sort=False).fillna(0))

            #|||||||UnitDoneEvent
                _UDE = []
                _UDE_P1 = []
                _UDE_P2 = []
                _UDE_P1_agg = []
                _UDE_P2_agg = []

                _UDE_e = []

                for event in match[1]['UnitDoneEvent']:
                    UDE_D_ = {}
                    UDE_D_['player'] = event.unit.owner
                    UDE_D_['second'] = event.second
                    UDE_D_['winner'] = match[0].winner.players[0]
                    UDE_D_['map_name'] = match[0].map_name
                    UDE_D_['utcsi'] = event.unit.owner.pick_race  + '_' + event.unit._type_class.str_id
                    _UDE_e.append(UDE_D_)

                _UDE_e_df = pd.DataFrame(_UDE_e)

                print('_UDE | ', match[0].player[1],' | ',match[0].winner,' | ',match[0].map_name)
                _UDE_e_df_1 = _UDE_e_df[_UDE_e_df['player'] == match[0].player[1]]
                _UDE_e_df_1_gd = pd.get_dummies(_UDE_e_df_1['utcsi'], prefix = 'utcsi').cumsum(axis = 0)
                _UDE_P1_ = pd.concat([_UDE_e_df_1, _UDE_e_df_1_gd], axis = 1, sort=False).drop(columns = 'utcsi').fillna(0)
                _UDE_P1_ = _UDE_P1_[~_UDE_P1_['second'].duplicated(keep='last')]
                _UDE_P1_S_m = max(_UDE_P1_['second'])
                _UDE_P1_S_ = pd.DataFrame(list(range(0,_UDE_P1_S_m)), columns = ['second'])
                _UDE_P1_merge = pd.merge(_UDE_P1_, _UDE_P1_S_, on = 'second', how = 'right')
                _UDE_P1_df = _UDE_P1_merge.sort_values(by = 'second')
                _UDE_P1_df.index = _UDE_P1_df['second']
                _UDE_P1_df_i = _UDE_P1_df.interpolate(method = 'values').ffill().bfill()

                print('_UDE | ', match[0].player[2],' | ',match[0].winner,' | ',match[0].map_name)
                _UDE_e_df_2 = _UDE_e_df[_UDE_e_df['player'] == match[0].player[2]]
                _UDE_e_df_2_gd = pd.get_dummies(_UDE_e_df_2['utcsi'], prefix = 'utcsi').cumsum(axis = 0)
                _UDE_P2_ = pd.concat([_UDE_e_df_2, _UDE_e_df_2_gd], axis = 1, sort=False).drop(columns = 'utcsi').fillna(0)
                _UDE_P2_ = _UDE_P2_[~_UDE_P2_['second'].duplicated(keep='last')]
                _UDE_P2_S_m = max(_UDE_P2_['second'])
                _UDE_P2_S_ = pd.DataFrame(list(range(0,_UDE_P2_S_m)), columns = ['second'])
                _UDE_P2_merge = pd.merge(_UDE_P2_, _UDE_P2_S_, on = 'second', how = 'right')
                _UDE_P2_df = _UDE_P2_merge.sort_values(by = 'second')
                _UDE_P2_df.index = _UDE_P2_df['second']
                _UDE_P2_df_i = _UDE_P2_df.interpolate(method = 'values').ffill().bfill()

                replays_UnitDoneEvent.append(pd.concat([_UDE_P1_df_i, _UDE_P2_df_i], axis = 0, sort=False).fillna(0))

            #|||||||UnitBornEvent
                _UBE = []
                _UBE_P1 = []
                _UBE_P2 = []
                _UBE_P1_agg = []
                _UBE_P2_agg = []

                _UBE_e = []
                for event in match[1]['UnitBornEvent']:
                    UBE_D_ = {}
                    UBE_D_['player'] = event.unit_controller
                    UBE_D_['second'] = event.second
                    if (event.unit_controller == match[0].player[1]) or (event.unit_controller == match[0].player[2]):
                        UBE_D_['utn'] = event.unit_controller.pick_race + '_' + event.unit_type_name
                    else:
                        UBE_D_['utn'] = event.unit_type_name
                    _UBE_e.append(UBE_D_)

                _UBE_e_df = pd.DataFrame(_UBE_e)

                print('_UBE | ', match[0].player[1],' | ',match[0].winner,' | ',match[0].map_name)
                _UBE_e_df_1 = _UBE_e_df[_UBE_e_df['player'] == match[0].player[1]]
                _UBE_e_df_1_gd = pd.get_dummies(_UBE_e_df_1['utn'], prefix = 'utn').cumsum(axis = 0)
                _UBE_P1_ = pd.concat([_UBE_e_df_1, _UBE_e_df_1_gd], axis = 1, sort=False).drop(columns = 'utn').fillna(0)
                _UBE_P1_ = _UBE_P1_[~_UBE_P1_['second'].duplicated(keep='last')]
                _UBE_P1_S_m = max(_UBE_P1_['second'])
                _UBE_P1_S_ = pd.DataFrame(list(range(0,_UBE_P1_S_m)), columns = ['second'])
                _UBE_P1_merge = pd.merge(_UBE_P1_, _UBE_P1_S_, on = 'second', how = 'right')
                _UBE_P1_df = _UBE_P1_merge.sort_values(by = 'second')
                _UBE_P1_df.index = _UBE_P1_df['second']
                _UBE_P1_df_i = _UBE_P1_df.interpolate(method = 'values').ffill()

                print('_UBE | ', match[0].player[2],' | ',match[0].winner,' | ',match[0].map_name)
                _UBE_e_df_2 = _UBE_e_df[_UBE_e_df['player'] == match[0].player[2]]
                _UBE_e_df_2_gd = pd.get_dummies(_UBE_e_df_2['utn'], prefix = 'utn').cumsum(axis = 0)
                _UBE_P2_ = pd.concat([_UBE_e_df_2, _UBE_e_df_2_gd], axis = 1, sort=False).drop(columns = 'utn').fillna(0)
                _UBE_P2_ = _UBE_P2_[~_UBE_P2_['second'].duplicated(keep='last')]
                _UBE_P2_S_m = max(_UBE_P2_['second'])
                _UBE_P2_S_ = pd.DataFrame(list(range(0,_UBE_P2_S_m)), columns = ['second'])
                _UBE_P2_merge = pd.merge(_UBE_P2_, _UBE_P2_S_, on = 'second', how = 'right')
                _UBE_P2_df = _UBE_P2_merge.sort_values(by = 'second')
                _UBE_P2_df.index = _UBE_P2_df['second']
                _UBE_P2_df_i = _UBE_P2_df.interpolate(method = 'values').ffill()

                replays_UnitBornEvent.append(pd.concat([_UBE_P1_df_i, _UBE_P2_df_i], axis = 0, sort=False).fillna(0))
            except:
                print(match[0])
                pass

        return replays_PlayerStatsEvent, replays_BasicCommandEvent, replays_TargetPointCommandEvent, replays_UnitBornEvent, replays_UnitDoneEvent

    replays_events = extract_replay_events(version_[_version])
    replays_PlayerStatsEvent, replays_BasicCommandEvent, replays_TargetPointCommandEvent, replays_UnitBornEvent, replays_UnitDoneEvent = ETL(replays_events)

    DOMAIN_BCE = []
    DOMAIN_TPC = []

    for match in range(0, len(replays_PlayerStatsEvent)):
        try:
            A0 = replays_PlayerStatsEvent[match][replays_PlayerStatsEvent[match]['player'] == replays_PlayerStatsEvent[match]['player'].unique()[0]]
            B0 = replays_PlayerStatsEvent[match][replays_PlayerStatsEvent[match]['player'] == replays_PlayerStatsEvent[match]['player'].unique()[1]]

            A0_col = ['PSE_' + col if col != 'second' else col for col in A0.columns]
            B0_col = ['PSE_' + col if col != 'second' else col for col in B0.columns]

            #import pdb; pdb.set_trace()

            A0.columns = A0_col
            B0.columns = B0_col

            A1 = replays_UnitBornEvent[match][replays_UnitBornEvent[match]['player'] == replays_UnitBornEvent[match]['player'].unique()[0]]
            B1 = replays_UnitBornEvent[match][replays_UnitBornEvent[match]['player'] == replays_UnitBornEvent[match]['player'].unique()[1]]

            A1_col = ['UBE_' + col if col != 'second' else col for col in A1.columns]
            B1_col = ['UBE_' + col if col != 'second' else col for col in B1.columns]

            A1.columns = A1_col
            B1.columns = B1_col

            A2 = replays_UnitDoneEvent[match][replays_UnitDoneEvent[match]['player'] == replays_UnitDoneEvent[match]['player'].unique()[0]]
            B2 = replays_UnitDoneEvent[match][replays_UnitDoneEvent[match]['player'] == replays_UnitDoneEvent[match]['player'].unique()[1]]

            A2_col = ['UDE_' + col if col != 'second' else col for col in A2.columns]
            B2_col = ['UDE_' + col if col != 'second' else col for col in B2.columns]

            A2.columns = A2_col
            B2.columns = B2_col

            t_A0 = replays_BasicCommandEvent[match][replays_BasicCommandEvent[match]['player'] == replays_BasicCommandEvent[match]['player'].unique()[0]]
            t_B0 = replays_BasicCommandEvent[match][replays_BasicCommandEvent[match]['player'] == replays_BasicCommandEvent[match]['player'].unique()[1]]

            t_A0_col = ['BCE_' + col if col != 'second' else col for col in t_A0.columns]
            t_B0_col = ['BCE_' + col if col != 'second' else col for col in t_B0.columns]

            t_A0.columns = t_A0_col
            t_B0.columns = t_B0_col

            t_A1 = replays_TargetPointCommandEvent[match][replays_TargetPointCommandEvent[match]['player'] == replays_TargetPointCommandEvent[match]['player'].unique()[0]]
            t_B1 = replays_TargetPointCommandEvent[match][replays_TargetPointCommandEvent[match]['player'] == replays_TargetPointCommandEvent[match]['player'].unique()[1]]

            t_A1_col = ['TPC_' + col if col != 'second' else col for col in t_A1.columns]
            t_B1_col = ['TPC_' + col if col != 'second' else col for col in t_B1.columns]

            t_A1.columns = t_A1_col
            t_B1.columns = t_B1_col

            A = [A0, A1, A2]
            B = [B0, B1, B2]

            A_t = [t_A0, t_A1]
            B_t = [t_B0, t_B1]

            df_final_A = reduce(lambda left,right: pd.merge(left,right,on='second'), A)
            df_final_B = reduce(lambda left,right: pd.merge(left,right,on='second'), B)

            #|||||||||Merge dataframes here.
            Q = pd.merge(df_final_A, A_t[0], on = 'second', how = 'right').interpolate(method = 'values').bfill()
            W = pd.merge(df_final_A, A_t[1], on = 'second', how = 'right').interpolate(method = 'values').bfill()
            E = pd.merge(df_final_B, B_t[0], on = 'second', how = 'right').interpolate(method = 'values').bfill()
            R = pd.merge(df_final_B, B_t[1], on = 'second', how = 'right').interpolate(method = 'values').bfill()

            DOMAIN_BCE.append(Q), DOMAIN_BCE.append(E)
            DOMAIN_TPC.append(W), DOMAIN_TPC.append(R)
        except:
            print(match)
            pass

    DOMAIN_BCE_c = pd.concat(DOMAIN_BCE, axis = 0, ignore_index = True)
    DOMAIN_TPC_c = pd.concat(DOMAIN_TPC, axis = 0, ignore_index = True)

    Relevent_Column_Names_BCE = ['UDE_map_name', 'BCE_player', 'UDE_winner',
    'PSE_food_made', 'PSE_food_used',
    'PSE_minerals_collection_rate', 'PSE_minerals_current',
    'second',
    'PSE_vespene_collection_rate', 'PSE_vespene_current',
    'PSE_workers_active_count',
    'UBE_utn_Terran_Marauder', 'UBE_utn_Terran_Marine', 'UBE_utn_Terran_Medivac', 'UBE_utn_Terran_Reaper', 'UBE_utn_Terran_SCV', 'UBE_utn_Terran_WidowMine',
    'UDE_utcsi_Terran_Barracks', 'UDE_utcsi_Terran_BarracksReactor', 'UDE_utcsi_Terran_BarracksTechLab', 'UDE_utcsi_Terran_Bunker',
    'UDE_utcsi_Terran_EngineeringBay', 'UDE_utcsi_Terran_Factory', 'UDE_utcsi_Terran_FactoryReactor', 'UDE_utcsi_Terran_OrbitalCommand',
    'UDE_utcsi_Terran_Refinery', 'UDE_utcsi_Terran_Starport', 'UDE_utcsi_Terran_StarportReactor', 'UDE_utcsi_Terran_SupplyDepot',
    'BCE_ability_name']

    Relevent_Column_Names_TPC = ['UDE_map_name', 'TPC_player', 'UDE_winner',
    'PSE_food_made', 'PSE_food_used',
    'PSE_minerals_collection_rate', 'PSE_minerals_current',
    'second',
    'PSE_vespene_collection_rate', 'PSE_vespene_current',
    'PSE_workers_active_count',
    'UBE_utn_Terran_Marauder', 'UBE_utn_Terran_Marine', 'UBE_utn_Terran_Medivac', 'UBE_utn_Terran_Reaper', 'UBE_utn_Terran_SCV', 'UBE_utn_Terran_WidowMine',
    'UDE_utcsi_Terran_Barracks', 'UDE_utcsi_Terran_BarracksReactor', 'UDE_utcsi_Terran_BarracksTechLab', 'UDE_utcsi_Terran_Bunker',
    'UDE_utcsi_Terran_EngineeringBay', 'UDE_utcsi_Terran_Factory', 'UDE_utcsi_Terran_FactoryReactor', 'UDE_utcsi_Terran_OrbitalCommand',
    'UDE_utcsi_Terran_Refinery', 'UDE_utcsi_Terran_Starport', 'UDE_utcsi_Terran_StarportReactor', 'UDE_utcsi_Terran_SupplyDepot',
    'TPC_ability_name', 'TPC_x', 'TPC_y']

    # Q = DOMAIN_BCE_c[DOMAIN_BCE_c['BCE_player'].apply(lambda x: x.pick_race) == 'Terran'].fillna(0)[Relevent_Column_Names_BCE]
    # Q_ = DOMAIN_TPC_c[DOMAIN_TPC_c['TPC_player'].apply(lambda x: x.pick_race) == 'Terran'].fillna(0)[Relevent_Column_Names_TPC]

    Q = DOMAIN_BCE_c[DOMAIN_BCE_c['BCE_player'].apply(lambda x: x.pick_race) == 'Terran'].fillna(0)
    Q_ = DOMAIN_TPC_c[DOMAIN_TPC_c['TPC_player'].apply(lambda x: x.pick_race) == 'Terran'].fillna(0)

    Q.to_csv('./matchup_/ZvT_BCE_c.csv')
    Q_.to_csv('./matchup_/ZvT_TPC_c.csv')
