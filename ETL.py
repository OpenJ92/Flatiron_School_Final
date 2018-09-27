import sc2reader
import pandas as pd
from sc2reader.engine.plugins import SelectionTracker, APMTracker
sc2reader.engine.register_plugin(SelectionTracker())
sc2reader.engine.register_plugin(APMTracker())
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

#https://github.com/GraylinKim/sc2reader/blob/master/sc2reader/events/tracker.py#L79
#https://github.com/GraylinKim/sc2reader/blob/master/sc2reader/events/game.py

# python-sc2
# vars(self.state.common)
#https://github.com/Dentosal/python-sc2/issues/33

#### GOALS for the weekend
#complete######### A: Load all replays into list
#complete######### B: Run through each replay and ask if a player in this match is Terran.
#complete######### C: Extract PlayerStatsEvent (carry out linear interpolation on each),
##########            BasicCommandEvent, player.untis??, player._____??, AttackEvents, etc.etc.
#complete######### D: Build sklearn sub-library for logistic, linear, ensamble, SVM techniques
#complete######### E: Apply simple ensamble ML to output BasicCommandEvents given a subset of PlayerStatsEvent
########## F: Create mappings from ML output (BasicCommandEvent) to python-sc2 functions

#### ML goals
########## B: Use assorted ensamble ML methods to carry our desicions
########## C: Use KNN, SVM, Logistic regression on scrapped data.
########## D: Think about how you might use the upcoming module in this project.

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#######ETL
load = False

replays = []
replays_with_terran_terran = []
replays_with_terran_zerg = []
replays_with_terran_protoss = []
replays_with_zerg_zerg = []
replays_with_zerg_protoss = []
replays_with_protoss_protoss = []

_path = '/Users/flatironschool/Desktop/PySC2/sc2reader/sc2_Replays/2018 WCS Valencia'
_day = ['/Day 3 - RO8', '/Day 2 - RO16', '/Day 1 - RO80']
_groupStage = ['Group Stage 1', 'Group Stage 2', 'Group Stage 3']
_group = ['Group ' + letter for letter in list(map(chr, range(65,91)))]

#Day_1
if load:
    D1_path = {}
    D1_path_ = {}
    for groupStage in _groupStage:
        for group in _group:
            try:
                D1_path[_path + _day[2] + '/' + groupStage + '/' + group] = os.listdir(_path + _day[2] + '/' + groupStage + '/' + group)
            except:
                pass
    for _paths in D1_path.keys():
        for match in D1_path[_paths]:
            if match == '.DS_Store':
                pass
            else:
                D1_path_[_paths + '/' + match] = os.listdir(_paths + '/' + match)
    for _paths in D1_path_.keys():
        for game in D1_path_[_paths]:
            print(_paths + '/' + game)
            replays.append(sc2reader.load_replay(_paths + '/' + game))

#Day_2
if load:
    D2_path = []
    for _paths in os.listdir(_path + _day[1]):
        if _paths == '.DS_Store':
            pass
        else:
            D2_path.append(_path + _day[1] + '/' + _paths)
    for _paths in D2_path:
        for game in os.listdir(_paths):
            print(_paths + '/' + game)
            replays.append(sc2reader.load_replay(_paths + '/' + game))

#Day_3
D3_path = []
for _paths in os.listdir(_path + _day[0]):
    if _paths == '.DS_Store':
        pass
    else:
        D3_path.append(_path + _day[0] + '/' + _paths)
for _paths in D3_path:
    for game in os.listdir(_paths):
        print(_paths + '/' + game)
        replays.append(sc2reader.load_replay(_paths + '/' + game))

#Match T-T, T-Z, T-P, Z-Z, Z-P, P-P
for replay_ in replays:
    if ((replay_.players[0].play_race == 'Terran') and (replay_.players[1].play_race == 'Zerg')) or ((replay_.players[0].play_race == 'Zerg') and (replay_.players[1].play_race == 'Terran')):
        replays_with_terran_zerg.append(replay_)
    if ((replay_.players[0].play_race == 'Terran') and (replay_.players[1].play_race == 'Protoss')) or (replay_.players[0].play_race == 'Protoss') and (replay_.players[1].play_race == 'Terran'):
        replays_with_terran_protoss.append(replay_)
    if ((replay_.players[0].play_race == 'Zerg') and (replay_.players[1].play_race == 'Protoss')) or (replay_.players[0].play_race == 'Protoss') and (replay_.players[1].play_race == 'Zerg'):
        replays_with_zerg_protoss.append(replay_)
    if (replay_.players[0].play_race == 'Zerg') and (replay_.players[1].play_race == 'Zerg'):
        replays_with_zerg_zerg.append(replay_)
    if (replay_.players[0].play_race == 'Protoss') and (replay_.players[1].play_race == 'Protoss'):
        replays_with_protoss_protoss.append(replay_)
    if (replay_.players[0].play_race == 'Terran') and (replay_.players[1].play_race == 'Terran'):
        replays_with_terran_terran.append(replay_)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# In [46]: replays_events[0][1].keys()
# Out[46]: dict_keys(['AddToControlGroupEvent', 'GetControlGroupEvent', 'PlayerLeaveEvent',
#                     'UnitDiedEvent', 'BasicCommandEvent'***, 'UserOptionsEvent', 'ProgressEvent',
#                     'SetControlGroupEvent', 'UpdateTargetPointCommandEvent', 'UnitInitEvent',
#                     'PlayerStatsEvent'***, 'TargetUnitCommandEvent', 'UnitTypeChangeEvent',
#                     'CameraEvent', 'ControlGroupEvent', 'UnitDoneEvent', 'SelectionEvent',
#                     'UpgradeCompleteEvent', 'UnitPositionsEvent', 'UpdateTargetUnitCommandEvent',
#                     'TargetPointCommandEvent'***, 'UnitBornEvent', 'PlayerSetupEvent', 'ChatEvent'])

# In [47]: replays_game_events[0][1].keys()
# Out[47]: dict_keys(['TargetUnitCommandEvent', 'UpdateTargetUnitCommandEvent', 'UserOptionsEvent',
#                     'SetControlGroupEvent', 'TargetPointCommandEvent', 'CameraEvent',
#                     'ControlGroupEvent', 'AddToControlGroupEvent', 'SelectionEvent',
#                     'UpdateTargetPointCommandEvent', 'GetControlGroupEvent', 'PlayerLeaveEvent', 'BasicCommandEvent'])

#replay.events
replay_events_name = [event.name for event in replays[0].events]
replay_events_name_unique = list(set(replay_events_name))
replay_game_events_name = [event.name for event in replays[0].game_events]
replay_game_events_name_unique = list(set(replay_game_events_name))

def replay_game_events_filter_by_name(name, replay_event_method):
    return [event for event in replay_event_method if event.name == name]

replays_game_events = []
replays_events = []
replays_tracker_events = []

for _replay in replays:
    print(_replay)
    replays_game_events.append((_replay, {name:replay_game_events_filter_by_name(name, _replay.game_events) for name in replay_game_events_name_unique}))
    replays_events.append((_replay, {name:replay_game_events_filter_by_name(name, _replay.events) for name in replay_events_name_unique}))
    replays_tracker_events.append((_replay, _replay.tracker_events))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def interpolate_PlayerStatsEvent(vars_e0, vars_e1):
    c_int = []
    c_dict = {}
    for int_ in range(0,10):
        for key in vars_e0.keys():
            if key not in ['player', 'stats', 'name']:
                c_dict[key] = vars_e0[key] + (int_/10)*(vars_e1[key] - vars_e0[key])
            else:
                c_dict[key] = vars_e0[key]
        c_int.append(c_dict)
        c_dict = {}
    return c_int

replays_PlayerStatsEvent = []
replays_BasicCommandEvent = []
replays_TargetPointCommandEvent = []
replays_UnitBornEvent = []
replays_UnitInitEvent = []
replays_UnitDiedEvent = []
replays_UnitDoneEvent = []

for match in replays_events:
    _PSE = []
    _PSE_0 = []
    _PSE_1 = []
    for event in range(0,len(match[1]['PlayerStatsEvent'])):
        if match[1]['PlayerStatsEvent'][event].player == match[0].player[1]:
            _PSE_0.append(match[1]['PlayerStatsEvent'][event])
        else:
            _PSE_1.append(match[1]['PlayerStatsEvent'][event])
    for event in range(0,len(_PSE_0) - 1):
        _PSE += interpolate_PlayerStatsEvent(vars(_PSE_0[event]),vars(_PSE_0[event + 1]))
    for event in range(0,len(_PSE_1) - 1):
        _PSE += interpolate_PlayerStatsEvent(vars(_PSE_1[event]),vars(_PSE_1[event + 1]))
    replays_PlayerStatsEvent.append(pd.DataFrame(_PSE))

# goal: construct list of dictionaries which reflect current army units. ie:
#         {'Marine_total': 10, 'Marauder_total': 5, 'SCV_total': 18, ect...} -- (7/10)*complete
#         {'Barracks': 3, 'Factory': 2, 'Starport': 1, ect...}

for match in replays_events:
    _UBE = []
    _UBE_P1 = []
    _UBE_P2 = []
    _UBE_P1_agg = []
    _UBE_P2_agg = []
    compressed_UBE_P1_ = []

####Domain

    for event in match[1]['UnitBornEvent']:
        if event.unit_controller == match[0].player[1]:
            UBE_D_P1 = {}
            UBE_D_P1['player'] = event.unit_controller
            UBE_D_P1['second'] = event.second
            if event.unit_type_name in UBE_D_P1.keys():
                UBE_D_P1[event.unit_type_name] += 1
            else:
                UBE_D_P1[event.unit_type_name] = 1
            _UBE_P1.append(UBE_D_P1)
        elif event.unit_controller == match[0].player[2]:
            UBE_D_P2 = {}
            UBE_D_P2['player'] = event.unit_controller
            UBE_D_P2['second'] = event.second
            if event.unit_type_name in UBE_D_P2.keys():
                UBE_D_P2[event.unit_type_name] += 1
            else:
                UBE_D_P2[event.unit_type_name] = 1
            _UBE_P2.append(UBE_D_P2)
    for agg_events in range(1, len(_UBE_P1)):
        _UBE_P1_agg.append(pd.DataFrame(_UBE_P1).fillna(0).iloc[:agg_events].sum(axis = 0))
    for agg_events in range(1, len(_UBE_P2)):
        _UBE_P2_agg.append(pd.DataFrame(_UBE_P2).fillna(0).iloc[:agg_events].sum(axis = 0))

    _UBE_P1_ = pd.concat([pd.DataFrame(_UBE_P1_agg).drop(columns = ['second']), pd.DataFrame(_UBE_P1)['second']], axis = 1).fillna(pd.DataFrame(_UBE_P1)['player'].unique()[0]).iloc[:-2]
    _UBE_P2_ = pd.concat([pd.DataFrame(_UBE_P2_agg).drop(columns = ['second']), pd.DataFrame(_UBE_P2)['second']], axis = 1).fillna(pd.DataFrame(_UBE_P2)['player'].unique()[0]).iloc[:-2]

    #Drop duplicates:
    print('_')
    _UBE_P1_ = _UBE_P1_[~_UBE_P1_['second'].duplicated(keep='last')]
    _UBE_P1_S_m = max(_UBE_P1_['second'])
    _UBE_P1_S_ = pd.DataFrame(list(range(0,_UBE_P1_S_m)), columns = ['second'])
    _UBE_P1_merge = pd.merge(_UBE_P1_, _UBE_P1_S_, on = 'second', how = 'right')
    _UBE_P1_df = _UBE_P1_merge.sort_values(by = 'second')
    _UBE_P1_df.index = _UBE_P1_df['second']
    _UBE_P1_df_i = _UBE_P1_df.interpolate().ffill()

    print('_')
    _UBE_P2_ = _UBE_P2_[~_UBE_P2_['second'].duplicated(keep='last')]
    _UBE_P2_S_m = max(_UBE_P2_['second'])
    _UBE_P2_S_ = pd.DataFrame(list(range(0,_UBE_P2_S_m)), columns = ['second'])
    _UBE_P2_merge = pd.merge(_UBE_P2_, _UBE_P2_S_, on = 'second', how = 'right')
    _UBE_P2_df = _UBE_P2_merge.sort_values(by = 'second')
    _UBE_P2_df.index = _UBE_P2_df['second']
    _UBE_P2_df_i = _UBE_P2_df.interpolate().ffill()

    replays_UnitBornEvent.append(pd.concat([_UBE_P1_df_i, _UBE_P2_df_i], axis = 0).fillna(0))

################UDE_ info here
########vars(vars(replays_UnitDoneEvent[10]['unit'].iloc[2])['_type_class'])
########vars(replays_UnitDoneEvent[10]['unit'].iloc[2])

##############Finish UDE then write Medium article

for match in replays_events:
    _UDE = []
    for event in match[1]['UnitDoneEvent']:
        _UDE.append(vars(event))
    replays_UnitDoneEvent.append(pd.DataFrame(_UDE))

############Range

for match in replays_events:
    _BCE = []
    for event in match[1]['BasicCommandEvent']:
        _BCE.append(vars(event))
    replays_BasicCommandEvent.append(pd.DataFrame(_BCE))

for match in replays_events:
    _TPE = []
    for event in match[1]['TargetPointCommandEvent']:
        _TPE.append(vars(event))
    replays_TargetPointCommandEvent.append(pd.DataFrame(_TPE))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

####Player Stats Event for one player
####Look to generalize for all matches
UIE_BCE_TPC_PlayerStatsEvent = []

for match in range(0, len(replays_PlayerStatsEvent)):
    for player in range(0,2):
        try:
            #grab unique players
            _PSE = replays_PlayerStatsEvent[match][replays_PlayerStatsEvent[match]['player'] == replays_PlayerStatsEvent[match]['player'].unique()[player]]
            #_UIE = replays_UnitInitEvent[match][replays_UnitInitEvent[match]['unit_controller'] == replays_PlayerStatsEvent[match]['player'].unique()[player]]
            _BCE = replays_BasicCommandEvent[match][replays_BasicCommandEvent[match]['player'] == replays_PlayerStatsEvent[match]['player'].unique()[player]]
            _TPC = replays_TargetPointCommandEvent[match][replays_TargetPointCommandEvent[match]['player'] == replays_PlayerStatsEvent[match]['player'].unique()[player]]
            UIE_BCE_TPC_PlayerStatsEvent.append([replays_PlayerStatsEvent[match]['player'].unique()[player], replays_PlayerStatsEvent[match]['player'].unique()[player - 1],pd.merge(_PSE, _UIE, on = 'second'), pd.merge(_PSE, _BCE, on = 'second'), pd.merge(_PSE, _TPC, on = 'second')])
        except:
            print(replays_PlayerStatsEvent[match]['player'].unique()[player])


#PSE_UIE = pd.concat([match[2] for match in UIE_BCE_TPC_PlayerStatsEvent], axis = 0)
PSE_BCE = pd.concat([match[2] for match in UIE_BCE_TPC_PlayerStatsEvent], axis = 0)
PSE_TPC = pd.concat([match[3] for match in UIE_BCE_TPC_PlayerStatsEvent], axis = 0)

# PSE_UIE.to_csv('./PSE_UIE.csv', index = False)
# PSE_BCE.to_csv('./PSE_BCE.csv', index = False)
# PSE_TPC.to_csv('./PSE_TPC.csv', index = False)
