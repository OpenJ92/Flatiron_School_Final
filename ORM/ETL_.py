from __init__ import db
import numpy as np
import os
from sqlalchemy import or_
import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker
sc2reader.engine.register_plugin(SelectionTracker())
sc2reader.engine.register_plugin(APMTracker())
from models import Participant, User, Game, PlayerStatsEvent, UnitBornEvent, \
                        UnitTypeChangeEvent, UpgradeCompleteEvent, UnitDoneEvent, \
                        BasicCommandEvent, TargetPointEvent, UnitDiedEvent, UnitInitEvent

# Evaluate new schema --

# User
    # has-one:
    # has-many: Participants
# Participant
    # has-one: Game, User
    # has-many: Events
# Game
    # has-one:
    # has-many: Participants
# Events
    # has-one: Participant
    # has-many:

# ORDER OF CONSTRUCTION:
    # 1) User -- if not yet constructed
    # 2) Game
    # 3) Participant(Game, User)
    # 4) [Event(Participant) for event in EVENTS]

# Game(1) - Participant (A) - User (A)
#             |
#           events ()

# Game(1) - Participant (B) - User (B)
#             |
#           events ()

#consider rebuilding 'construct_objects(replay_file)' function

def construct_objects(replay_file, pro = False):
    try:
        replay = sc2reader.load_replay(replay_file)

        game = db.session.query(Game).filter_by(name = str(replay.date) + '_' + replay.players[0].play_race + ' v ' + replay.players[1].play_race + '_' + replay.players[0].name + ' v ' + replay.players[1].name).first()

        if game != None:
            return None

        game = Game(name = str(replay.date) + '_' + replay.players[0].play_race + ' v ' + replay.players[1].play_race + '_' + replay.players[0].name + ' v ' + replay.players[1].name,
                    map = replay.map_name,
                    game_winner = replay.winner.players[0].name,
                    start_time = replay.start_time,
                    end_time = replay.end_time,
                    category = replay.category,
                    expansion = replay.expansion,
                    time_zone = replay.time_zone
                    )

        userOne = db.session.query(User).filter_by(name = replay.players[0].name).first()
        userTwo = db.session.query(User).filter_by(name = replay.players[1].name).first()

        if userOne == None:
            userOne = User(name = replay.players[0].name, region = replay.players[0].region, subregion = replay.players[0].subregion)

        if userTwo == None:
            userTwo = User(name = replay.players[1].name, region = replay.players[1].region, subregion = replay.players[1].subregion)

        users = [userOne, userTwo]

        if replay.players[0].is_human:
            highest_league_playerOne = replay.players[0].highest_league
            avg_apm_playerOne = replay.players[0].avg_apm
            if pro:
                highest_league_playerOne = 20
        else:
            highest_league_playerOne = -1
            avg_apm_playerOne = -1

        if replay.players[1].is_human:
            highest_league_playerTwo = replay.players[1].highest_league
            avg_apm_playerTwo = replay.players[1].avg_apm
            if pro:
                highest_league_playerTwo = 20
        else:
            highest_league_playerTwo = -1
            avg_apm_playerTwo = -1

        participantOne = Participant(user = [users[0]],
                                     game = [game],
                                     name = userOne.name,
                                     league = highest_league_playerOne,
                                     scaled_rating = replay.raw_data['replay.initData']['user_initial_data'][0]['scaled_rating'],
                                     playrace = replay.players[0].play_race,
                                     avg_apm = avg_apm_playerOne,
                                     winner = userOne.name == replay.winner.players[0].name
                                     )

        participantTwo = Participant(user = [users[1]],
                                     game = [game],
                                     name = userTwo.name,
                                     league = highest_league_playerTwo,
                                     scaled_rating = replay.raw_data['replay.initData']['user_initial_data'][1]['scaled_rating'],
                                     playrace = replay.players[0].play_race,
                                     avg_apm = avg_apm_playerTwo,
                                     winner = userTwo.name == replay.winner.players[0].name
                                     )

        participants = [participantOne, participantTwo]
        events = replay.events
        participantOne_events = []
        participantTwo_events = []

        for event in events:
            try:
                if event.name == 'PlayerStatsEvent':
                    if event.player.name == participants[0].name:
                        participantOne_events.append(PlayerStatsEvent(participant = participants[0],
                                                                 name = event.name,
                                                                 second = event.second,
                                                                 minerals_current = event.minerals_current,
                                                                 vespene_current = event.vespene_current,
                                                                 minerals_collection_rate = event.minerals_collection_rate,
                                                                 vespene_collection_rate = event.vespene_collection_rate,
                                                                 workers_active_count = event.workers_active_count,
                                                                 minerals_used_in_progress_army = event.minerals_used_in_progress_army,
                                                                 minerals_used_in_progress_economy = event.minerals_used_in_progress_economy,
                                                                 minerals_used_in_progress_technology = event.minerals_used_in_progress_technology,
                                                                 minerals_used_in_progress = event.minerals_used_in_progress,
                                                                 vespene_used_in_progress_army = event.vespene_used_in_progress_army,
                                                                 vespene_used_in_progress_economy = event.vespene_used_in_progress_economy,
                                                                 vespene_used_in_progress_technology = event.vespene_used_in_progress_technology,
                                                                 vespene_used_in_progress = event.vespene_used_in_progress,
                                                                 resources_used_in_progress = event.resources_used_in_progress,
                                                                 minerals_used_current_army = event.minerals_used_current_army,
                                                                 minerals_used_current_economy = event.minerals_used_current_economy,
                                                                 minerals_used_current_technology = event.minerals_used_current_technology,
                                                                 minerals_used_current = event.minerals_used_current,
                                                                 vespene_used_current_army = event.vespene_used_current_army,
                                                                 vespene_used_current_economy = event.vespene_used_current_economy,
                                                                 vespene_used_current_technology = event.vespene_used_current_technology,
                                                                 vespene_used_current = event.vespene_used_current,
                                                                 resources_used_current = event.resources_used_current,
                                                                 minerals_lost_army = event.minerals_lost_army,
                                                                 minerals_lost_economy = event.minerals_lost_economy,
                                                                 minerals_lost_technology = event.minerals_lost_technology,
                                                                 minerals_lost = event.minerals_lost,
                                                                 vespene_lost_army = event.vespene_lost_army,
                                                                 vespene_lost_economy = event.vespene_lost_economy,
                                                                 vespene_lost_technology = event.vespene_lost_technology,
                                                                 vespene_lost = event.vespene_lost,
                                                                 resources_lost = event.resources_lost,
                                                                 minerals_killed_army = event.minerals_killed_army,
                                                                 minerals_killed_economy = event.minerals_killed_economy,
                                                                 minerals_killed_technology = event.minerals_killed_technology,
                                                                 minerals_killed = event.minerals_killed,
                                                                 vespene_killed_army = event.vespene_killed_army,
                                                                 vespene_killed_economy = event.vespene_killed_economy,
                                                                 vespene_killed_technology = event.vespene_killed_technology,
                                                                 vespene_killed = event.vespene_killed,
                                                                 resources_killed = event.resources_killed,
                                                                 food_used = event.food_used,
                                                                 food_made = event.food_made,
                                                                 minerals_used_active_forces = event.minerals_used_active_forces,
                                                                 vespene_used_active_forces = event.vespene_used_active_forces,
                                                                 ff_minerals_lost_army = event.ff_minerals_lost_army,
                                                                 ff_minerals_lost_economy = event.ff_minerals_lost_economy,
                                                                 ff_minerals_lost_technology = event.ff_minerals_lost_technology,
                                                                 ff_vespene_lost_army = event.ff_vespene_lost_army,
                                                                 ff_vespene_lost_economy = event.ff_vespene_lost_economy,
                                                                 ff_vespene_lost_technology = event.ff_vespene_lost_technology
                                                                 ))
                    else:
                        participantTwo_events.append(PlayerStatsEvent(participant = participants[1],
                                                                 name = event.name,
                                                                 second = event.second,
                                                                 minerals_current = event.minerals_current,
                                                                 vespene_current = event.vespene_current,
                                                                 minerals_collection_rate = event.minerals_collection_rate,
                                                                 vespene_collection_rate = event.vespene_collection_rate,
                                                                 workers_active_count = event.workers_active_count,
                                                                 minerals_used_in_progress_army = event.minerals_used_in_progress_army,
                                                                 minerals_used_in_progress_economy = event.minerals_used_in_progress_economy,
                                                                 minerals_used_in_progress_technology = event.minerals_used_in_progress_technology,
                                                                 minerals_used_in_progress = event.minerals_used_in_progress,
                                                                 vespene_used_in_progress_army = event.vespene_used_in_progress_army,
                                                                 vespene_used_in_progress_economy = event.vespene_used_in_progress_economy,
                                                                 vespene_used_in_progress_technology = event.vespene_used_in_progress_technology,
                                                                 vespene_used_in_progress = event.vespene_used_in_progress,
                                                                 resources_used_in_progress = event.resources_used_in_progress,
                                                                 minerals_used_current_army = event.minerals_used_current_army,
                                                                 minerals_used_current_economy = event.minerals_used_current_economy,
                                                                 minerals_used_current_technology = event.minerals_used_current_technology,
                                                                 minerals_used_current = event.minerals_used_current,
                                                                 vespene_used_current_army = event.vespene_used_current_army,
                                                                 vespene_used_current_economy = event.vespene_used_current_economy,
                                                                 vespene_used_current_technology = event.vespene_used_current_technology,
                                                                 vespene_used_current = event.vespene_used_current,
                                                                 resources_used_current = event.resources_used_current,
                                                                 minerals_lost_army = event.minerals_lost_army,
                                                                 minerals_lost_economy = event.minerals_lost_economy,
                                                                 minerals_lost_technology = event.minerals_lost_technology,
                                                                 minerals_lost = event.minerals_lost,
                                                                 vespene_lost_army = event.vespene_lost_army,
                                                                 vespene_lost_economy = event.vespene_lost_economy,
                                                                 vespene_lost_technology = event.vespene_lost_technology,
                                                                 vespene_lost = event.vespene_lost,
                                                                 resources_lost = event.resources_lost,
                                                                 minerals_killed_army = event.minerals_killed_army,
                                                                 minerals_killed_economy = event.minerals_killed_economy,
                                                                 minerals_killed_technology = event.minerals_killed_technology,
                                                                 minerals_killed = event.minerals_killed,
                                                                 vespene_killed_army = event.vespene_killed_army,
                                                                 vespene_killed_economy = event.vespene_killed_economy,
                                                                 vespene_killed_technology = event.vespene_killed_technology,
                                                                 vespene_killed = event.vespene_killed,
                                                                 resources_killed = event.resources_killed,
                                                                 food_used = event.food_used,
                                                                 food_made = event.food_made,
                                                                 minerals_used_active_forces = event.minerals_used_active_forces,
                                                                 vespene_used_active_forces = event.vespene_used_active_forces,
                                                                 ff_minerals_lost_army = event.ff_minerals_lost_army,
                                                                 ff_minerals_lost_economy = event.ff_minerals_lost_economy,
                                                                 ff_minerals_lost_technology = event.ff_minerals_lost_technology,
                                                                 ff_vespene_lost_army = event.ff_vespene_lost_army,
                                                                 ff_vespene_lost_economy = event.ff_vespene_lost_economy,
                                                                 ff_vespene_lost_technology = event.ff_vespene_lost_technology
                                                                 ))

                elif event.name == 'UnitBornEvent':
                    if event.unit_controller.name == participants[0].name:
                        participantOne_events.append(UnitBornEvent(participant = participants[0],
                                                              name = event.name,
                                                              second = event.second,
                                                              unit_type_name = event.unit_type_name,
                                                              loc_x = event.x,
                                                              loc_y = event.y
                                                              ))
                    else:
                        participantTwo_events.append(UnitBornEvent(participant = participants[1],
                                                              name = event.name,
                                                              second = event.second,
                                                              unit_type_name = event.unit_type_name,
                                                              loc_x = event.x,
                                                              loc_y = event.y
                                                              ))

                elif event.name == 'UnitTypeChangeEvent':
                    if event.unit.owner.name == participants[0].name:
                        participantOne_events.append(UnitTypeChangeEvent(participant = participants[0],
                                                                    name = event.name,
                                                                    second = event.second,
                                                                    unit = event.unit.name,
                                                                    unit_type_name = event.unit_type_name
                                                                    ))
                    else:
                        participantTwo_events.append(UnitTypeChangeEvent(participant = participants[1],
                                                                    name = event.name,
                                                                    second = event.second,
                                                                    unit = event.unit.name,
                                                                    unit_type_name = event.unit_type_name
                                                                    ))

                elif event.name == 'UpgradeCompleteEvent':
                    if event.player.name == participants[0].name:
                        participantOne_events.append(UpgradeCompleteEvent(participant = participants[0],
                                                                     name = event.name,
                                                                     second = event.second,
                                                                     upgrade_type_name = event.upgrade_type_name
                                                                     ))
                    else:
                        participantTwo_events.append(UpgradeCompleteEvent(participant = participants[1],
                                                                     name = event.name,
                                                                     second = event.second,
                                                                     upgrade_type_name = event.upgrade_type_name
                                                                     ))

                elif event.name == 'UnitInitEvent':
                    if event.unit_controller.name == participants[0].name:
                        participantOne_events.append(UnitInitEvent(participant = participants[0],
                                                              name = event.name,
                                                              second = event.second,
                                                              unit_type_name = event.unit_type_name,
                                                              loc_x = event.x,
                                                              loc_y = event.y
                                                              ))
                    else:
                        participantOne_events.append(UnitInitEvent(participant = participants[1],
                                                              name = event.name,
                                                              second = event.second,
                                                              unit_type_name = event.unit_type_name,
                                                              loc_x = event.x,
                                                              loc_y = event.y
                                                              ))

                elif event.name == 'UnitDoneEvent':
                    if event.unit.owner.name == participants[0].name:
                        participantOne_events.append(UnitDoneEvent(participant = participants[0],
                                                              name = event.name,
                                                              second = event.second,
                                                              unit = event.unit.name
                                                              ))
                    else:
                        participantTwo_events.append(UnitDoneEvent(participant = participants[1],
                                                              name = event.name,
                                                              second = event.second,
                                                              unit = event.unit.name
                                                              ))

                elif event.name == 'BasicCommandEvent':
                    if event.player.name == participants[0].name:
                        participantOne_events.append(BasicCommandEvent(participant = participants[0],
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  ability_name = event.ability_name
                                                                  ))
                    else:
                        participantTwo_events.append(BasicCommandEvent(participant = participants[1],
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  ability_name = event.ability_name
                                                                  ))

                elif event.name == 'TargetPointCommandEvent':
                    if event.player.name == participants[0].name:
                        participantOne_events.append(TargetPointEvent(participant = participants[0],
                                                                 name = event.name,
                                                                 second = event.second,
                                                                 ability_name = event.ability_name,
                                                                 loc_x = event.x,
                                                                 loc_y = event.y
                                                                 ))
                    else:
                        participantTwo_events.append(TargetPointEvent(participant = participants[1],
                                                                 name = event.name,
                                                                 second = event.second,
                                                                 ability_name = event.ability_name,
                                                                 loc_x = event.x,
                                                                 loc_y = event.y
                                                                 ))
                elif event.name == 'UnitDiedEvent':
                    if event.killing_player.name == participants[0].name:
                        # if event.killer_pid != None:
                        #     import pdb; pdb.set_trace()
                        if event.killing_unit.owner.name != event.unit.owner.name:
                            participantOne_events.append(UnitDiedEvent(participant = participants[0],
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  killing_unit = event.killing_unit.name,
                                                                  unit = event.unit.name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))
        #participantOne_events.append(UnitDiedEvent(participant = [participants[0]], killing_participant = [participants[1]], name = event.name, second = event.second, killing_unit = event.killing_unit.name, unit = event.unit.name, loc_x = event.x, loc_y = event.y))
        #UnitDiedEvent(participant = participants[0], killing_participant = participants[1], game = game, name = event.name, second = event.second, killing_unit = event.killing_unit.name, unit = event.unit.name, loc_x = event.x, loc_y = event.y)
                    else:
                        # if event.killer_pid != None:
                        #     import pdb; pdb.set_trace()
                        if event.killing_unit.owner.name != event.unit.owner.name:
                            participantTwo_events.append(UnitDiedEvent(participant = participants[1],
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  killing_unit = event.killing_unit.name,
                                                                  unit = event.unit.name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))

            except Exception as e:
                pass
                #print(e, event.name)
                # if event.name == 'PlayerStatsEvent':
                #     print(event.name, event.player)
                # elif event.name == 'UnitBornEvent':
                #     print(participants[0].name, participants[1].name, game, event.name, event.unit_controller, event.unit_type_name, event.second, event.x, event.y)

        #import pdb; pdb.set_trace()
        db.session.add_all(participantOne_events + participantTwo_events + participants + [game] + users)
        db.session.commit()

        return [participantOne, participantTwo, userOne, userTwo, game]
    except Exception as e:
        pass
        #print(e)
        #print('replay: failed to load')




if False:
    def construct_objects_v0(replay_file, pro = False):
        try:
            #import pdb; pdb.set_trace()
            replay = sc2reader.load_replay(replay_file)
            game = db.session.query(Game).filter_by(name = str(replay.date) + '_' + replay.players[0].play_race + ' v ' + replay.players[1].play_race).first()

            if game != None:
                #print('Game: exists ------------------')
                return None

            playerOne = db.session.query(User).filter_by(name = replay.players[0].name).first()
            playerTwo = db.session.query(User).filter_by(name = replay.players[1].name).first()

            if playerOne == None:
                playerOne = User(name = replay.players[0].name, region = replay.players[0].region, subregion = replay.players[0].subregion)

            if playerTwo == None:
                playerTwo = User(name = replay.players[1].name, region = replay.players[1].region, subregion = replay.players[1].subregion)

            if replay.players[0].is_human:
                highest_league_playerOne = replay.players[0].highest_league
                avg_apm_playerOne = replay.players[0].avg_apm
                if pro:
                    highest_league_playerOne = 20
            else:
                highest_league_playerOne = -1
                avg_apm_playerOne = -1

            if replay.players[1].is_human:
                highest_league_playerTwo = replay.players[1].highest_league
                avg_apm_playerTwo = replay.players[1].avg_apm
                if pro:
                    highest_league_playerTwo = 20
            else:
                highest_league_playerTwo = -1
                avg_apm_playerTwo = -1

            users = [playerOne, playerTwo]
            players = [Player(name = player.name, region = player.region, subregion = player.subregion) for player in replay.players]
            game = Game(players = players, name = str(replay.date) + '_' + replay.players[0].play_race + ' v ' + replay.players[1].play_race,
                        map = replay.map_name,
                        playerOne_name = players[0].name,
                        playerTwo_name = players[1].name,
                        playerOne_league = highest_league_playerOne,
                        playerTwo_league = highest_league_playerTwo,
                        playerOne_playrace = replay.players[0].play_race,
                        playerTwo_playrace = replay.players[1].play_race,
                        playerOne_avg_apm = avg_apm_playerOne,
                        playerTwo_avg_apm = avg_apm_playerTwo,
                        game_winner = replay.winner.players[0].name,
                        date = replay.date,
                        end_time = replay.end_time,
                        category = replay.category,
                        expansion = replay.expansion,
                        time_zone = replay.time_zone
                        )

            participants = [Participant(game = game, user = user[0])] #build out
            events = replay.events
            playerOne_events = []
            playerTwo_events = []

            for event in events:
                try:
                    if event.name == 'PlayerStatsEvent':
                        if event.player.name == players[0].name:
                            playerOne_events.append(PlayerStatsEvent(player = players[0], game = game,
                                                                     name = event.name,
                                                                     second = event.second,
                                                                     minerals_current = event.minerals_current,
                                                                     vespene_current = event.vespene_current,
                                                                     minerals_collection_rate = event.minerals_collection_rate,
                                                                     vespene_collection_rate = event.vespene_collection_rate,
                                                                     workers_active_count = event.workers_active_count,
                                                                     minerals_used_in_progress_army = event.minerals_used_in_progress_army,
                                                                     minerals_used_in_progress_economy = event.minerals_used_in_progress_economy,
                                                                     minerals_used_in_progress_technology = event.minerals_used_in_progress_technology,
                                                                     minerals_used_in_progress = event.minerals_used_in_progress,
                                                                     vespene_used_in_progress_army = event.vespene_used_in_progress_army,
                                                                     vespene_used_in_progress_economy = event.vespene_used_in_progress_economy,
                                                                     vespene_used_in_progress_technology = event.vespene_used_in_progress_technology,
                                                                     vespene_used_in_progress = event.vespene_used_in_progress,
                                                                     resources_used_in_progress = event.resources_used_in_progress,
                                                                     minerals_used_current_army = event.minerals_used_current_army,
                                                                     minerals_used_current_economy = event.minerals_used_current_economy,
                                                                     minerals_used_current_technology = event.minerals_used_current_technology,
                                                                     minerals_used_current = event.minerals_used_current,
                                                                     vespene_used_current_army = event.vespene_used_current_army,
                                                                     vespene_used_current_economy = event.vespene_used_current_economy,
                                                                     vespene_used_current_technology = event.vespene_used_current_technology,
                                                                     vespene_used_current = event.vespene_used_current,
                                                                     resources_used_current = event.resources_used_current,
                                                                     minerals_lost_army = event.minerals_lost_army,
                                                                     minerals_lost_economy = event.minerals_lost_economy,
                                                                     minerals_lost_technology = event.minerals_lost_technology,
                                                                     minerals_lost = event.minerals_lost,
                                                                     vespene_lost_army = event.vespene_lost_army,
                                                                     vespene_lost_economy = event.vespene_lost_economy,
                                                                     vespene_lost_technology = event.vespene_lost_technology,
                                                                     vespene_lost = event.vespene_lost,
                                                                     resources_lost = event.resources_lost,
                                                                     minerals_killed_army = event.minerals_killed_army,
                                                                     minerals_killed_economy = event.minerals_killed_economy,
                                                                     minerals_killed_technology = event.minerals_killed_technology,
                                                                     minerals_killed = event.minerals_killed,
                                                                     vespene_killed_army = event.vespene_killed_army,
                                                                     vespene_killed_economy = event.vespene_killed_economy,
                                                                     vespene_killed_technology = event.vespene_killed_technology,
                                                                     vespene_killed = event.vespene_killed,
                                                                     resources_killed = event.resources_killed,
                                                                     food_used = event.food_used,
                                                                     food_made = event.food_made,
                                                                     minerals_used_active_forces = event.minerals_used_active_forces,
                                                                     vespene_used_active_forces = event.vespene_used_active_forces,
                                                                     ff_minerals_lost_army = event.ff_minerals_lost_army,
                                                                     ff_minerals_lost_economy = event.ff_minerals_lost_economy,
                                                                     ff_minerals_lost_technology = event.ff_minerals_lost_technology,
                                                                     ff_vespene_lost_army = event.ff_vespene_lost_army,
                                                                     ff_vespene_lost_economy = event.ff_vespene_lost_economy,
                                                                     ff_vespene_lost_technology = event.ff_vespene_lost_technology
                                                                     ))
                        else:
                            playerTwo_events.append(PlayerStatsEvent(player = players[1], game = game,
                                                                     name = event.name,
                                                                     second = event.second,
                                                                     minerals_current = event.minerals_current,
                                                                     vespene_current = event.vespene_current,
                                                                     minerals_collection_rate = event.minerals_collection_rate,
                                                                     vespene_collection_rate = event.vespene_collection_rate,
                                                                     workers_active_count = event.workers_active_count,
                                                                     minerals_used_in_progress_army = event.minerals_used_in_progress_army,
                                                                     minerals_used_in_progress_economy = event.minerals_used_in_progress_economy,
                                                                     minerals_used_in_progress_technology = event.minerals_used_in_progress_technology,
                                                                     minerals_used_in_progress = event.minerals_used_in_progress,
                                                                     vespene_used_in_progress_army = event.vespene_used_in_progress_army,
                                                                     vespene_used_in_progress_economy = event.vespene_used_in_progress_economy,
                                                                     vespene_used_in_progress_technology = event.vespene_used_in_progress_technology,
                                                                     vespene_used_in_progress = event.vespene_used_in_progress,
                                                                     resources_used_in_progress = event.resources_used_in_progress,
                                                                     minerals_used_current_army = event.minerals_used_current_army,
                                                                     minerals_used_current_economy = event.minerals_used_current_economy,
                                                                     minerals_used_current_technology = event.minerals_used_current_technology,
                                                                     minerals_used_current = event.minerals_used_current,
                                                                     vespene_used_current_army = event.vespene_used_current_army,
                                                                     vespene_used_current_economy = event.vespene_used_current_economy,
                                                                     vespene_used_current_technology = event.vespene_used_current_technology,
                                                                     vespene_used_current = event.vespene_used_current,
                                                                     resources_used_current = event.resources_used_current,
                                                                     minerals_lost_army = event.minerals_lost_army,
                                                                     minerals_lost_economy = event.minerals_lost_economy,
                                                                     minerals_lost_technology = event.minerals_lost_technology,
                                                                     minerals_lost = event.minerals_lost,
                                                                     vespene_lost_army = event.vespene_lost_army,
                                                                     vespene_lost_economy = event.vespene_lost_economy,
                                                                     vespene_lost_technology = event.vespene_lost_technology,
                                                                     vespene_lost = event.vespene_lost,
                                                                     resources_lost = event.resources_lost,
                                                                     minerals_killed_army = event.minerals_killed_army,
                                                                     minerals_killed_economy = event.minerals_killed_economy,
                                                                     minerals_killed_technology = event.minerals_killed_technology,
                                                                     minerals_killed = event.minerals_killed,
                                                                     vespene_killed_army = event.vespene_killed_army,
                                                                     vespene_killed_economy = event.vespene_killed_economy,
                                                                     vespene_killed_technology = event.vespene_killed_technology,
                                                                     vespene_killed = event.vespene_killed,
                                                                     resources_killed = event.resources_killed,
                                                                     food_used = event.food_used,
                                                                     food_made = event.food_made,
                                                                     minerals_used_active_forces = event.minerals_used_active_forces,
                                                                     vespene_used_active_forces = event.vespene_used_active_forces,
                                                                     ff_minerals_lost_army = event.ff_minerals_lost_army,
                                                                     ff_minerals_lost_economy = event.ff_minerals_lost_economy,
                                                                     ff_minerals_lost_technology = event.ff_minerals_lost_technology,
                                                                     ff_vespene_lost_army = event.ff_vespene_lost_army,
                                                                     ff_vespene_lost_economy = event.ff_vespene_lost_economy,
                                                                     ff_vespene_lost_technology = event.ff_vespene_lost_technology
                                                                     ))

                    elif event.name == 'UnitBornEvent':
                        if event.unit_controller.name == players[0].name:
                            playerOne_events.append(UnitBornEvent(player = players[0], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  unit_type_name = event.unit_type_name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))
                        else:
                            playerTwo_events.append(UnitBornEvent(player = players[1], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  unit_type_name = event.unit_type_name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))
                    #######Note: Both player and killing player are being populated with both Participants
                    elif event.name == 'UnitDiedEvent':
                        if event.unit.owner.name == players[0].name:
                            #import pdb; pdb.set_trace()
                            playerOne_events.append(UnitDiedEvent(player = [players[0]], killing_player = [players[1]], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  killing_unit = event.killing_unit.name,
                                                                  unit = event.unit.name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))
            #UnitDiedEvent(player = players[0], killing_player = players[1], game = game, name = event.name, second = event.second, killing_unit = event.killing_unit.name, unit = event.unit.name, loc_x = event.x, loc_y = event.y)
                        else:
                            #import pdb; pdb.set_trace()
                            playerTwo_events.append(UnitDiedEvent(player = [players[1]], killing_player = [players[0]], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  killing_unit = event.killing_unit.name,
                                                                  unit = event.unit.name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))

                    elif event.name == 'UnitTypeChangeEvent':
                        if event.unit.owner.name == players[0].name:
                            playerOne_events.append(UnitTypeChangeEvent(player = players[0], game = game,
                                                                        name = event.name,
                                                                        second = event.second,
                                                                        unit = event.unit.name,
                                                                        unit_type_name = event.unit_type_name
                                                                        ))
                        else:
                            playerTwo_events.append(UnitTypeChangeEvent(player = players[1], game = game,
                                                                        name = event.name,
                                                                        second = event.second,
                                                                        unit = event.unit.name,
                                                                        unit_type_name = event.unit_type_name
                                                                        ))

                    elif event.name == 'UpgradeCompleteEvent':
                        if event.player.name == players[0].name:
                            playerOne_events.append(UpgradeCompleteEvent(player = players[0], game = game,
                                                                         name = event.name,
                                                                         second = event.second,
                                                                         upgrade_type_name = event.upgrade_type_name
                                                                         ))
                        else:
                            playerTwo_events.append(UpgradeCompleteEvent(player = players[1], game = game,
                                                                         name = event.name,
                                                                         second = event.second,
                                                                         upgrade_type_name = event.upgrade_type_name
                                                                         ))

                    elif event.name == 'UnitInitEvent':
                        if event.unit_controller.name == players[0].name:
                            playerOne_events.append(UnitInitEvent(player = players[0], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  unit_type_name = event.unit_type_name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))
                        else:
                            playerOne_events.append(UnitInitEvent(player = players[1], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  unit_type_name = event.unit_type_name,
                                                                  loc_x = event.x,
                                                                  loc_y = event.y
                                                                  ))

                    elif event.name == 'UnitDoneEvent':
                        if event.unit.owner.name == players[0].name:
                            playerOne_events.append(UnitDoneEvent(player = players[0], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  unit = event.unit.name
                                                                  ))
                        else:
                            playerTwo_events.append(UnitDoneEvent(player = players[1], game = game,
                                                                  name = event.name,
                                                                  second = event.second,
                                                                  unit = event.unit.name
                                                                  ))

                    elif event.name == 'BasicCommandEvent':
                        if event.player.name == players[0].name:
                            playerOne_events.append(BasicCommandEvent(player = players[0], game = game,
                                                                      name = event.name,
                                                                      second = event.second,
                                                                      ability_name = event.ability_name
                                                                      ))
                        else:
                            playerTwo_events.append(BasicCommandEvent(player = players[1], game = game,
                                                                      name = event.name,
                                                                      second = event.second,
                                                                      ability_name = event.ability_name
                                                                      ))

                    elif event.name == 'TargetPointCommandEvent':
                        if event.player.name == players[0].name:
                            playerOne_events.append(TargetPointEvent(player = players[0], game = game,
                                                                     name = event.name,
                                                                     second = event.second,
                                                                     ability_name = event.ability_name,
                                                                     loc_x = event.x,
                                                                     loc_y = event.y
                                                                     ))
                        else:
                            playerTwo_events.append(TargetPointEvent(player = players[1], game = game,
                                                                     name = event.name,
                                                                     second = event.second,
                                                                     ability_name = event.ability_name,
                                                                     loc_x = event.x,
                                                                     loc_y = event.y
                                                                     ))
                except:
                    if event.name == 'PlayerStatsEvent':
                        print(event.name, event.player)
                    elif event.name == 'UnitBornEvent':
                        print(players[0].name, players[1].name, game, event.name, event.unit_controller, event.unit_type_name, event.second, event.x, event.y)
                    elif event.name == 'UnitDiedEvent':
                        print(players[0].name, players[1].name, game, event.name, event.second, str(event.killing_unit), event.unit, event.x, event.y)

            db.session.add_all(playerOne_events + playerTwo_events + players + [game])
            db.session.commit()

            return players, game, playerOne_events, playerTwo_events
        except:
            pass
            #print('replay: failed to load')

def get_replays(league, numSamples = None, random = False):
    if random:
        return np.random.choice(np.array(['../sc2_Replays/' + league +  '/' + league + '_replays/' + link for link in os.listdir('../sc2_Replays/' + league +  '/' + league + '_replays') if '(' not in link]), numSamples)
    else:
        if numSamples != None:
            return ['../sc2_Replays/' + league +  '/' + league + '_replays/' + link for link in os.listdir('../sc2_Replays/' + league +  '/' + league + '_replays')][-1 * numSamples:]
        else:
            return ['../sc2_Replays/' + league +  '/' + league + '_replays/' + link for link in os.listdir('../sc2_Replays/' + league +  '/' + league + '_replays')]

def get_professional_replays(matchup):
    return ['../sc2_Replays/STReplays'+ matchup + '/' + file for file in os.listdir('../sc2_Replays/STReplays' + matchup)]


#Build recursive exception function that catches broken replay files and continues running
#on the next file in the directory.

counter = 0
for replay in get_replays('Gold', numsamples = None, random = False):
    try:
        construct_objects(replay)
        counter += 1
    except Exception as e:
        print(e)
        print(counter)

# [construct_objects(replay) for replay in get_replays('Bronze', 2000)]
# print('Bronze Complete____________________________________________________')
# [construct_objects(replay) for replay in get_replays('Silver', 2000)]
# print('Silver Complete____________________________________________________')
# [construct_objects(replay) for replay in get_replays('Gold', 2000)]
# print('Gold Complete____________________________________________________')
# [construct_objects(replay) for replay in get_replays('Platinum', 2000)]
# print('Platinum Complete____________________________________________________')
# [construct_objects(replay) for replay in get_replays('Diamond', 2000)]
# print('Diamond Complete____________________________________________________')
# [construct_objects(replay) for replay in get_replays('Master', 2000)]
# print('Master Complete____________________________________________________')
# [construct_objects(replay) for replay in get_replays('Grand_Master', 2000)]
# print('Grand_Master Complete____________________________________________________')
# [construct_objects(replay, pro = True) for replay in get_professional_replays('TvT')]
# print('TvT Complete____________________________________________________')
# [construct_objects(replay, pro = True) for replay in get_professional_replays('PvT')]
# print('PvT Complete____________________________________________________')
# [construct_objects(replay, pro = True) for replay in get_professional_replays('ZvT')]
# print('ZvT Complete____________________________________________________')
# [construct_objects(replay, pro = True) for replay in get_professional_replays('PvP')]
# print('PvP Complete____________________________________________________')
# [construct_objects(replay, pro = True) for replay in get_professional_replays('ZvP')]
# print('ZvP Complete____________________________________________________')
# [construct_objects(replay, pro = True) for replay in get_professional_replays('ZvZ')]
# print('ZvZ Complete____________________________________________________')
