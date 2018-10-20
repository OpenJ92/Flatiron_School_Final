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
