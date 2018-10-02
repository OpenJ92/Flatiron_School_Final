#from ORM.__init__ import db
import pandas as pd
from sqlalchemy import and_, or_
from models import *


print('enter routes.py')
# Player, Game, PlayerStatsEvent, UnitBornEvent, UnitDiedEvent, \
#                         UnitTypeChangeEvent, UpgradeCompleteEvent, UnitDoneEvent, \
#                         BasicCommandEvent, TargetPointEvent, Player_Games, Player_UDiE

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


def query():
    return db.session.query(Game)

def filter_by_Player_name(player):
    return db.session.query(Player).filter_by(name = player).first()

def filter_by_Player_region(region):
    return db.session.query(Player).filter_by(region = region).all()

def filter_Game_by_id(id):
    return db.session.query(Game).filter_by(id = id).first()

def filter_Game_by_name(name):
    return db.session.query(Game).filter_by(name = name).first()

def filter_by_Game_playerOneTwo_name(players):
    return db.session.query(Game).filter_by(or_(and_(playerOne_name == players[0],playerTwo_name == players[1]),and_(playerOne_name == players[1],playerTwo_name == players[0])))

def filter_by_Game_map_name(map_name):
    return db.session.query(Game).filter_by(map = map_name).all()

def filter_by_Game_category(category):
    return db.session.query(Game).filter_by(category = category).all()

def filter_by_Game_highest_league(highest_league):
    return db.session.query(Game).filter(or_(Game.playerOne_league == highest_league, Game.playerTwo_league == highest_league)).all()

def filter_by_Game_not_highest_league(highest_league):
    return db.session.query(Game).filter(or_(Game.playerOne_league != highest_league, Game.playerTwo_league != highest_league)).all()

def filter_by_Game_matchup(r):
    return db.session.query(Game).filter(or_(and_(Game.playerOne_playrace == r[0], Game.playerTwo_playrace == r[1]), and_(Game.playerOne_playrace == r[1], Game.playerTwo_playrace == r[0]))).all()

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||DataFrame from query
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def event_df_from_Game_PSE_highestLeague(value):
    query = filter_by_Game_highest_league(value)
    events = [game.events_PSE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_PSE_map(value):
    query = filter_by_Game_map_name(value)
    events = [game.events_PSE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_PSE_matchup(value):
    query = filter_by_Game_matchup(value)
    events = [game.events_PSE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_BCE_highestLeague(value):
    query = filter_by_Game_highest_league(value)
    events = [game.events_BCE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_BCE_map(value):
    query = filter_by_Game_map_name(value)
    events = [game.events_BCE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_BCE_matchup(value):
    query = filter_by_Game_matchup(value)
    events = [game.events_BCE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_UBE_highestLeague(value):
    query = filter_by_Game_highest_league(value)
    events = [game.events_UBE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_UBE_map(value):
    query = filter_by_Game_map_name(value)
    events = [game.events_UBE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_UBE_matchup(value):
    query = filter_by_Game_matchup(value)
    events = [game.events_UBE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_TPE_highestLeague(value):
    query = filter_by_Game_highest_league(value)
    events = [game.events_TPE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_TPE_map(value):
    query = filter_by_Game_map_name(value)
    events = [game.events_TPE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

def event_df_from_Game_TPE_matchup(value):
    query = filter_by_Game_matchup(value)
    events = [game.events_TPE for game in query]
    return pd.DataFrame([vars(event) for game in events for event in game])

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Manipulate Dataframe
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def PCA_UBE_hl(value):
    df = event_df_from_Game_UBE_highestLeague(value)
    unique_games = df['game_id'].unique()
    df_gd = pd.get_dummies(df['unit_type_name'], prefix = 'UDE_')
    df_gd_c = pd.concat([df, df_gd], axis = 1, sort = False)
    list_games = [[df_gd_c[(df_gd_c['game_id'] == game) & (df_gd_c['player_id'] == df_gd_c['player_id'].unique()[0])],
                    df_gd_c[(df_gd_c['game_id'] == game) & (df_gd_c['player_id'] == df_gd_c['player_id'].unique()[1])]]
                        for game in unique_games]
    relevent_columns = [col for col in list_games[0][0] if 'UDE' in col]
    return list_games, relevent_columns

def PCA_UBE_matchup(value):
    df = event_df_from_Game_UBE_matchup(value)
    unique_games = df['game_id'].unique()
    df_gd = pd.get_dummies(df['unit_type_name'], prefix = 'UDE_')
    df_gd_c = pd.concat([df, df_gd], axis = 1, sort = False)
    list_games = [[df_gd_c[(df_gd_c['game_id'] == game) & (df_gd_c['player_id'] == df_gd_c['player_id'].unique()[0])],
                    df_gd_c[(df_gd_c['game_id'] == game) & (df_gd_c['player_id'] == df_gd_c['player_id'].unique()[1])]]
                        for game in unique_games]
    relevent_columns = [col for col in list_games[0][0] if 'UDE' in col]
    return list_games, relevent_columns

def PCA_UBE_map(value):
    df = event_df_from_Game_UBE_map(value)
    unique_games = df['game_id'].unique()
    df_gd = pd.get_dummies(df['unit_type_name'], prefix = 'UDE_').cumsum()
    df_gd_c = pd.concat([df, df_gd], axis = 1, sort = False)
    list_games = [[df_gd_c[(df_gd_c['game_id'] == game) & (df_gd_c['player_id'] == df_gd_c['player_id'].unique()[0])],
                    df_gd_c[(df_gd_c['game_id'] == game) & (df_gd_c['player_id'] == df_gd_c['player_id'].unique()[1])]]
                        for game in unique_games]
    relevent_columns = [col for col in list_games[0][0] if 'UDE' in col]
    return list_games, relevent_columns

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Manipulate Dataframe
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def get_PSE_label():
    return [{'label': key, 'value': key} for key in vars(db.session.query(PlayerStatsEvent).first()).keys()]
def get_Player_names():
    return [{'label': player.name, 'value': player.name} for player in db.session.query(Player).all()]
def get_Game_names():
    return [{'label': game.name, 'value': game.name} for game in db.session.query(Game).all()]
def get_Game_maps():
    return [{'label': game, 'value': game} for game in set([game.map for game in db.session.query(Game).all()])]

print('exit routes.py')
