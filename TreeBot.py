import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import SCV, COMMANDCENTER, SUPPLYDEPOT, REFINERY, BARRACKS, MARINE, TECHLAB,  \
MARAUDER, FACTORY, HELLION, STARPORT, MEDIVAC, CYCLONE, REAPER, VIKINGFIGHTER, SIEGETANK, BANSHEE
import random
import numpy as np

class TreeBot(sc2.BotAI):
    def __init__(self):
        UNIT_ = {'Cyclone': 0, 'Marine': 0, 'Medivac': 0, 'SiegeTank': 0, 'Banshee': 0, 'Marauder': 0, 'SCV': 12}
        #import RandomForestClassifier

        pass

    async def on_step(self, iteration):
        
        obs = list(np.array([self.state.common.food_cap, self.state.common.food_used, self.state.common.food_workers]))
        print(obs)

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||Action Space Macro-E|||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    async def TrainCyclone(self):
        try:
            UNIT_['Cyclone'] += 1
            await self.do(self.units(FACTORY).ready.noqueue.random.train(CYCLONE))
        except:
            UNIT_['Cyclone'] -= 1
    async def TrainMarine(self):
        try:
            UNIT_['Marine'] += 1
            await self.do(self.units(BARRACKS).ready.noqueue.random.train(MARINE))
        except:
            UNIT_['Marine'] -= 1
    async def TrainMedivac(self):
        try:
            UNIT_['Medivac'] += 1
            await self.do(self.units(STARPORT).ready.noqueue.random.train(MEDIVAC))
        except:
            UNIT_['Medivac'] -= 1
            pass
    async def BuildSiegeTank(self):
        try:
            UNIT_['SiegeTank'] += 1
            await self.do(self.units(FACTORY).ready.noqueue.random.train(SIEGETANK))
        except:
            UNIT_['SiegeTank'] -= 1
            pass
    async def TrainBanshee(self):
        try:
            UNIT_['Banshee'] += 1
            await self.do(self.units(STARPORT).ready.noqueue.random.train(BANSHEE))
        except:
            UNIT_['Banshee'] -= 1
    async def TrainMarauder(self):
        try:
            UNIT_['Marauder'] += 1
            await self.do(self.units(BARRACKS).ready.noqueue.random.train(MARAUDER))
        except:
            UNIT_['Marauder'] -= 1
    async def TrainSCV(self):
        try:
            UNIT_['SCV'] += 1
            await self.do(self.units(COMMANDCENTER).ready.noqueue.random.train(SCV))
        except:
            UNIT_['SCV'] -= 1


    async def BuildCommandCenter(self):
        try:
            pass
        except:
            pass
    async def BuildEngineeringBay(self):
        try:
            pass
        except:
            pass
    async def BuildFactory(self):
        try:
            pass
        except:
            pass
    async def BuildBarracks(self):
        try:
            pass
        except:
            pass
    async def BuildStarport(self):
        try:
            pass
        except:
            pass

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||Action Space Micro-E|||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    async def attack(self):
        try:
            pass
        except:
            pass
    async def patrol(self):
        try:
            pass
        except:
            pass

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||Default-Commands|||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

run_game(maps.get('AbyssalReefLE'), [Bot(Race.Terran, TreeBot()),Computer(Race.Zerg, Difficulty.Hard)], realtime = False)
