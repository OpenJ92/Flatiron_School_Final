#from ORM.__init__ import db
import pandas as pd
from sqlalchemy import and_, or_, create_engine
from models import *


print('enter routes.py')
# Player, Game, PlayerStatsEvent, UnitBornEvent, UnitDiedEvent, \
#                         UnitTypeChangeEvent, UpgradeCompleteEvent, UnitDoneEvent, \
#                         BasicCommandEvent, TargetPointEvent, Player_Games, Player_UDiE

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Raw SQL Calls
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def random_():
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
         A = con.execute("""SELECT * FROM participants_users as UP
                            WHERE UP.user_id = (SELECT id FROM users WHERE name = 'ODSTJacob')""").fetchall()
         B = con.execute("""SELECT DISTINCT name FROM participants""").fetchall()
         C = con.execute("""SELECT DISTINCT map FROM games""").fetchall()
         D = con.execute("""SELECT * FROM users
                            WHERE name like 'O%' and region = 'us'""").fetchall()
         E = con.execute("""SELECT * FROM
                            (SELECT * FROM users as u
                            INNER JOIN participants_users as up
                            ON u.id = up.user_id) AS uup
                            INNER JOIN participants
                            ON participants.id = uup.participant_id""").fetchall()
         F = con.execute("""SELECT * FROM users as u
                            INNER JOIN
                            participants_users as up
                            ON u.id = up.user_id""").fetchall()

def SQL_call_with_input(input_):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute("""SELECT * FROM users WHERE name = %s""" % input_).fetchall()

def get_PlayerStatsEvents(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute(""" SELECT * FROM playerstatsevents AS PSE WHERE PSE.participant_id = %s""" % participant.id).fetchall()

def get_UnitBornEvent(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute(""" SELECT * FROM unitbornevents AS UBE WHERE UBE.participant_id = %s""" % participant.id).fetchall()

def get_UnitDiedEvent(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute(""" SELECT * FROM unitdiedevents AS UDE WHERE UDE.participant_id = %s""" % participant.id).fetchall()

def get_UnitTypeChangeEvent(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute("""SELECT * FROM unittypechangeevents AS UTCE WHERE UTCE.participant_id = %s""" % participant.id).fetchall()

def get_UnitInitEvent(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute("""SELECT * FROM unitinitevent AS UIE WHERE UIE.participant_id = %s""" % participant.id).fetchall()

def get_UnitDoneEvent(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute("""SELECT * FROM unitdoneevent AS UDE WHERE UDE.participant_id = %s""" % participant.id).fetchall()

def get_BasicCommandEvent(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute("""SELECT * FROM basiccommandevent AS BCE WHERE BCE.participant_id = %s""" % participant.id).fetchall()

def get_TargetPointEvent(participant):
    engine = create_engine('sqlite:///replays.db')
    with engine.connect() as con:
        return con.execute("""SELECT * FROM targetpointevent AS TPE WHERE TPE.participant_id = %s""" % participant.id).fetchall()


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||untested
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def participant_Name(name):
    return db.session.query(Participant).filter(Participant.name == name).all()

def participant_Playrace(playrace, sample_size):
    # generate a random sample from query.
    return db.session.query(Participant).filter(Participant.playrace == playrace).all()

def participant_scaledRating(lowerBound, upperBound):
    return db.session.query(Participant).filter((Participant.scaled_rating >= lowerBound)&(Participant.scaled_rating <= upperBound)).all()

def participant_league(league):
    return db.session.query(Participant).filter(Participant.league == league).all()

def participant_winner(winner):
    # generate a random sample from query.
    return db.session.query(Participant).filter(Participant.winner == winner).all()

def participant_winner_playrace(winner, playrace):
    return db.session.query(Participant).filter((Participant.winner == winner)&(Participant.playrace == playrace))

def participants_via_users(name):
    user = db.session.query(User).filter(User.name == name).first()
    return [participant for participant in user.participants]

def game_Map(map):
    # generate a random sample from query.
    return db.session.query(Game).filter(Game.map == map).all()

def game_Time(upperBound, lowerBound):
    #Note: upper and lower bound are in units of seconds.
    return db.session.query(Game).filter((Game.end_time - Game.start_time > lowerBound)&(Game.end_time - Game.start_time < upperBound))
def game_Expansion(expansion):
    # generate a random sample from query.
    return db.session.query(Game).filter(Game.expansion == expansion).all()

def unitbornevent_Name(unit_type_name):
    # generate a random sample from query
    return db.session.query(UnitBornEvent).filter(UnitBornEvent.unit_type_name == unit_type_name).all()

def unitdiedevent_KillingUnit(killing_unit):
    # generate a random sample from query
    return db.session.query(UnitDiedEvent).filter(UnitDiedEvent.killing_unit == killing_unit).all()

def unitdiedevent_Unit(unit):
    # generate a random sample from query
    return db.session.query(UnitDiedEvent).filter(UnitDiedEvent.unit == unit).all()

def unitdiedevent_location(location, radius):
    # generate a random sample from query
    return db.session.query(UnitDiedEvent).filter(radius**2 >= (UnitDiedEvent.loc_x - location[0])**2 + (UnitDiedEvent.loc_y - location[1])**2)

def upgradecompleteevent_Name(upgrade_type_name):
    return db.session.query(UpgradeCompleteEvent).filter(UpgradeCompleteEvent.upgrade_type_name == upgrade_type_name).all()

def unitinitevent_unit(unit_type_name):
    return db.session.query(UnitInitEvent).filter(UnitInitEvent.unit_type_name == unit_type_name).all()

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
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
