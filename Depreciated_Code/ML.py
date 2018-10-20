import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

PSE_BCE = pd.read_csv('PSE_BCE.csv')
PSE_TPC = pd.read_csv('PSE_TPC.csv')
PSE_UIE = pd.read_csv('PSE_UIE.csv')

PSE_BCE_t = PSE_BCE[PSE_BCE['player_x'].str.contains('Terran')]
PSE_TPC_t = PSE_TPC[PSE_TPC['player_x'].str.contains('Terran')]
PSE_UIE_t = PSE_UIE[PSE_UIE['player'].str.contains('Terran')]

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#Consider using UnitBornEvent to construct CurrentUnitDomain ie {'Marine': 10, 'SCV': 20, etc...} for training
#economy isn't enough. Maybe that's all one needs  ______ETL.py   COMPLETE -> See Sc.py and MLSc.py
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

PSE_BCE_t_SCV = PSE_BCE_t[PSE_BCE_t['ability_name'] == "TrainSCV"]
PSE_BCE_t_t1 = PSE_BCE_t[(PSE_BCE_t['ability_name'] == 'TrainCyclone') | (PSE_BCE_t['ability_name'] == 'TrainMarine') | (PSE_BCE_t['ability_name'] == 'TrainMedivac')]
PSE_BCE_t_t2 = PSE_BCE_t[(PSE_BCE_t['ability_name'] == 'BuildSiegeTank') | (PSE_BCE_t['ability_name'] == 'TrainBanshee') |(PSE_BCE_t['ability_name'] == 'TrainMarauder')]
PSE_BCE_t_Upgrade = PSE_BCE_t[(PSE_BCE_t['ability_name'] == 'UpgradeTerranInfantryWeapons1') | (PSE_BCE_t['ability_name'] == 'UpgradeTerranInfantryArmor1') | (PSE_BCE_t['ability_name'] == 'UpgradeTerranInfantryWeapons2') | (PSE_BCE_t['ability_name'] == 'UpgradeTerranInfantryArmor2') | (PSE_BCE_t['ability_name'] == 'UpgradeTerranInfantryWeapons3') | (PSE_BCE_t['ability_name'] == 'UpgradeTerranInfantryArmor3')]
#Consider scoping down PSE_TCP_t_Build
PSE_TCP_t_Build = PSE_TPC_t[(PSE_TPC_t['ability_name'] == 'BuildArmory') | (PSE_TPC_t['ability_name'] ==  'BuildBarracks') |
                         (PSE_TPC_t['ability_name'] == 'BuildCommandCenter') | (PSE_TPC_t['ability_name'] == 'BuildEngineeringBay') |
                         (PSE_TPC_t['ability_name'] == 'BuildFactory') | (PSE_TPC_t['ability_name'] == 'BuildStarport')]
PSE_TCP_t_Attack = PSE_TPC_t[(PSE_TPC_t['ability_name'] == 'Attack') | (PSE_TPC_t['ability_name'] == 'Move') | (PSE_TPC_t['ability_name'] == 'Patrol')]

PSE_BCE_t_X = PSE_BCE_t[['food_made', 'food_used', 'workers_active_count']]
PSE_TCP_t_X = PSE_TPC_t[['food_made', 'food_used', 'workers_active_count']]
PSE_UIE_t_X = PSE_UIE_t[['food_made', 'food_used', 'workers_active_count']]
PSE_BCE_t_y = PSE_BCE_t['ability_name']
PSE_TCP_t_y = PSE_TPC_t['ability_name']
PSE_UIE_t_y = PSE_UIE_t['unit_type_name']

PSE_BCE_t_SCV_X = PSE_BCE_t_SCV[['food_made', 'food_used', 'workers_active_count', 'second']]
PSE_BCE_t_SCV_y = PSE_BCE_t_SCV['ability_name']
PSE_BCE_t_t1_X = PSE_BCE_t_t1[['food_made', 'food_used', 'workers_active_count', 'second']]
PSE_BCE_t_t1_y = PSE_BCE_t_t1['ability_name']
PSE_BCE_t_t2_X = PSE_BCE_t_t2[['food_made', 'food_used', 'workers_active_count', 'second']]
PSE_BCE_t_t2_y = PSE_BCE_t_t2['ability_name']
PSE_BCE_t_Upgrade_X = PSE_BCE_t_Upgrade[['food_made', 'food_used', 'workers_active_count', 'second']]
PSE_BCE_t_Upgrade_y = PSE_BCE_t_Upgrade['ability_name']
PSE_TCP_t_Build_X = PSE_TCP_t_Build[['food_made', 'food_used', 'workers_active_count', 'second']]
PSE_TCP_t_Build_y = PSE_TCP_t_Build['ability_name']
PSE_TCP_t_Attack_X = PSE_TCP_t_Attack[['food_made', 'food_used', 'workers_active_count', 'second']]
PSE_TCP_t_Attack_y = PSE_TCP_t_Attack['ability_name']

def SMOTE_(X_train, y_train):
    SMOTE = pd.concat([X_train, y_train], axis = 1)
    SMOTE_list = [SMOTE[SMOTE[col] == 1] for col in y_train.columns]

    max_sample = max(list(map(lambda x: len(x), SMOTE_list)))
    SMOTE_list_synthetic = []

    for i in SMOTE_list:
        synthetic_data = []
        for j in range(len(i), max_sample):
            a = i.iloc[np.random.randint(len(i))]
            b = i.iloc[np.random.randint(len(i))]
            synthetic_data.append(np.array(a + (np.random.random_sample()*(b - a))))
        try:
            t = pd.DataFrame(synthetic_data)
            t.columns = i.columns
            SMOTE_list_synthetic.append(pd.concat([i,t]))
        except:
            SMOTE_list_synthetic.append(i)

    SMOTE_complete = pd.concat(SMOTE_list_synthetic)
    SMOTE_complete_X_train = SMOTE_complete[X_train.columns]
    SMOTE_complete_y_train = SMOTE_complete[y_train.columns]

    return SMOTE_complete_X_train, SMOTE_complete_y_train

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||RandomForestClassifier
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    X_train_BCE, X_test_BCE, y_train_BCE, y_test_BCE = train_test_split(PSE_BCE_t_X, pd.get_dummies(PSE_BCE_t_y), test_size=0.2, random_state=42)
    X_train_TCP, X_test_TCP, y_train_TCP, y_test_TCP = train_test_split(PSE_TCP_t_X, pd.get_dummies(PSE_TCP_t_y), test_size=0.2, random_state=42)
    X_train_UIE, X_test_UIE, y_train_UIE, y_test_UIE = train_test_split(PSE_UIE_t_X, pd.get_dummies(PSE_UIE_t_y), test_size=0.2, random_state=42)

    y_train_BCE_Unit = y_train_BCE[['BuildHellion', 'BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder',
       'TrainMarine', 'TrainMedivac', 'TrainRaven', 'TrainReaper', 'TrainSCV',
       'TrainViking']]
    y_test_BCE_Unit = y_test_BCE[['BuildHellion', 'BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder',
       'TrainMarine', 'TrainMedivac', 'TrainRaven', 'TrainReaper', 'TrainSCV',
       'TrainViking']]
    y_train_BCE_Upgrade = y_train_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_test_BCE_Upgrade = y_test_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_train_TCP_Build = y_train_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_test_TCP_Build = y_test_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_train_TCP_Attack = y_train_TCP[['Attack','Move','Patrol']]
    y_test_TCP_Attack = y_test_TCP[['Attack','Move','Patrol']]

    PSE_BCE_t_Unit_clf = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5)
    PSE_BCE_t_Upgrade_clf = RandomForestClassifier(criterion = 'entropy', n_estimators = 20, max_depth = 5)
    PSE_TCP_t_Build_clf = RandomForestClassifier(criterion = 'entropy', n_estimators = 20, max_depth = 5)
    PSE_TCP_t_Attack_clf = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5)

    PSE_BCE_t_Unit_clf.fit(X_train_BCE, y_train_BCE_Unit)
    PSE_BCE_t_Upgrade_clf.fit(X_train_BCE, y_train_BCE_Upgrade)
    PSE_TCP_t_Build_clf.fit(X_train_TCP, y_train_TCP_Build)
    PSE_TCP_t_Attack_clf.fit(X_train_TCP, y_train_TCP_Attack)

    target_ = [y_train_BCE_Unit, y_train_BCE_Upgrade, y_train_TCP_Build, y_train_TCP_Attack]
    for target in target_:
        print('--------------')
        print(target.shape)
        print('--------------')
        for i in target.keys():
            print(i, sum(target[i]))

    cm_Unit = confusion_matrix(y_test_BCE_Unit.values.argmax(axis=1), PSE_BCE_t_Unit_clf.predict(X_test_BCE).argmax(axis=1))
    cm_Upgrade = confusion_matrix(y_test_BCE_Upgrade.values.argmax(axis=1), PSE_BCE_t_Upgrade_clf.predict(X_test_BCE).argmax(axis=1))
    cm_Build = confusion_matrix(y_test_TCP_Build.values.argmax(axis=1), PSE_TCP_t_Build_clf.predict(X_test_TCP).argmax(axis=1))
    cm_Attack = confusion_matrix(y_test_TCP_Attack.values.argmax(axis=1), PSE_TCP_t_Attack_clf.predict(X_test_TCP).argmax(axis=1))

    print('RFC - Unit Accuracy: ',PSE_BCE_t_Unit_clf.score(X_test_BCE, y_test_BCE_Unit))
    print('RFC - Upgrade Accuracy: ',PSE_BCE_t_Upgrade_clf.score(X_test_BCE, y_test_BCE_Upgrade))
    print('RFC - Build Accuracy',PSE_TCP_t_Build_clf.score(X_test_TCP, y_test_TCP_Build))
    print('RFC - Attack Accuracy',PSE_TCP_t_Attack_clf.score(X_test_TCP, y_test_TCP_Attack))

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||RandomForestClassifier_
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    X_train_BCE, X_test_BCE, y_train_BCE, y_test_BCE = train_test_split(PSE_BCE_t_X, pd.get_dummies(PSE_BCE_t_y), test_size=0.2, random_state=42)
    X_train_TCP, X_test_TCP, y_train_TCP, y_test_TCP = train_test_split(PSE_TCP_t_X, pd.get_dummies(PSE_TCP_t_y), test_size=0.2, random_state=42)
    X_train_UIE, X_test_UIE, y_train_UIE, y_test_UIE = train_test_split(PSE_UIE_t_X, pd.get_dummies(PSE_UIE_t_y), test_size=0.2, random_state=42)

    y_train_BCE_Unit_SCV = y_train_BCE[['TrainSCV']]
    y_test_BCE_Unit_SCV = y_test_BCE[['TrainSCV']]
    y_train_BCE_Unit_t1 = y_train_BCE[['BuildHellion','TrainCyclone','TrainMarine','TrainMedivac',
    'TrainRaven', 'TrainReaper','TrainViking']]
    y_test_BCE_Unit_t1 = y_test_BCE[['BuildHellion','TrainCyclone','TrainMarine','TrainMedivac',
    'TrainRaven', 'TrainReaper','TrainViking']]
    y_train_BCE_Unit_t2 = y_train_BCE[['BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder']]
    y_test_BCE_Unit_t2 = y_test_BCE[['BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder']]
    y_train_BCE_Upgrade = y_train_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_test_BCE_Upgrade = y_test_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_train_TCP_Build = y_train_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_test_TCP_Build = y_test_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_train_TCP_Attack = y_train_TCP[['Attack','Move','Patrol']]
    y_test_TCP_Attack = y_test_TCP[['Attack','Move','Patrol']]

    PSE_BCE_t_SCV_clf_2 = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5)
    PSE_BCE_t_t1_clf_2 = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5)
    PSE_BCE_t_t2_clf_2 = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5)
    PSE_BCE_t_Upgrade_clf_2 = RandomForestClassifier(criterion = 'entropy', n_estimators = 20, max_depth = 5)
    PSE_TCP_t_Build_clf_2 = RandomForestClassifier(criterion = 'entropy', n_estimators = 20, max_depth = 5)
    PSE_TCP_t_Attack_clf_2 = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5)

    PSE_BCE_t_SCV_clf_2.fit(X_train_BCE, y_train_BCE_Unit_SCV)
    PSE_BCE_t_t1_clf_2.fit(X_train_BCE, y_train_BCE_Unit_t1)
    PSE_BCE_t_t2_clf_2.fit(X_train_BCE, y_train_BCE_Unit_t2)
    PSE_BCE_t_Upgrade_clf_2.fit(X_train_BCE, y_train_BCE_Upgrade)
    PSE_TCP_t_Build_clf_2.fit(X_train_TCP, y_train_TCP_Build)
    PSE_TCP_t_Attack_clf_2.fit(X_train_TCP, y_train_TCP_Attack)

    target_ = [y_train_BCE_Unit_SCV, y_train_BCE_Unit_t1, y_train_BCE_Unit_t2, y_train_BCE_Upgrade, y_train_TCP_Build, y_train_TCP_Attack]
    for target in target_:
        print('--------------')
        print(target.shape)
        print('--------------')
        for i in target.keys():
            print(i, sum(target[i]))
    #####Fix
    cm_SCV = confusion_matrix(y_test_BCE_Unit_SCV, PSE_BCE_t_SCV_clf_2.predict(X_test_BCE))
    cm_t1 = confusion_matrix(y_test_BCE_Unit_t1.values.argmax(axis=1), PSE_BCE_t_t1_clf_2.predict(X_test_BCE).argmax(axis=1))
    cm_t2 = confusion_matrix(y_test_BCE_Unit_t2.values.argmax(axis=1), PSE_BCE_t_t2_clf_2.predict(X_test_BCE).argmax(axis=1))
    cm_Upgrade = confusion_matrix(y_test_BCE_Upgrade.values.argmax(axis=1), PSE_BCE_t_Upgrade_clf_2.predict(X_test_BCE).argmax(axis=1))
    cm_Build = confusion_matrix(y_test_TCP_Build.values.argmax(axis=1), PSE_TCP_t_Build_clf_2.predict(X_test_TCP).argmax(axis=1))
    cm_Attack = confusion_matrix(y_test_TCP_Attack.values.argmax(axis=1), PSE_TCP_t_Attack_clf_2.predict(X_test_TCP).argmax(axis=1))

    print('clf_2 - SCV Accuracy: ',PSE_BCE_t_SCV_clf_2.score(X_test_BCE, y_test_BCE_Unit_SCV))
    print('clf_2 - t1 Accuracy: ',PSE_BCE_t_t1_clf_2.score(X_test_BCE, y_test_BCE_Unit_t1))
    print('clf_2 - t2 Accuracy: ',PSE_BCE_t_t2_clf_2.score(X_test_BCE, y_test_BCE_Unit_t2))
    print('clf_2 - Upgrade Accuracy: ',PSE_BCE_t_Upgrade_clf_2.score(X_test_BCE, y_test_BCE_Upgrade))
    print('clf_2 - Build Accuracy: ',PSE_TCP_t_Build_clf_2.score(X_test_TCP, y_test_TCP_Build))
    print('clf_2 - Attack Accuracy: ',PSE_TCP_t_Attack_clf_2.score(X_test_TCP, y_test_TCP_Attack))

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||RandomForestClassifier_3
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if True:
    X_train_BCE_SCV, X_test_BCE_SCV, y_train_BCE_SCV, y_test_BCE_SCV = train_test_split(PSE_BCE_t_SCV_X, pd.get_dummies(PSE_BCE_t_SCV_y), test_size=0.2, random_state=42)
    X_train_BCE_t1, X_test_BCE_t1, y_train_BCE_t1, y_test_BCE_t1 = train_test_split(PSE_BCE_t_t1_X, pd.get_dummies(PSE_BCE_t_t1_y), test_size=0.2, random_state=42)
    X_train_BCE_t2, X_test_BCE_t2, y_train_BCE_t2, y_test_BCE_t2 = train_test_split(PSE_BCE_t_t2_X, pd.get_dummies(PSE_BCE_t_t2_y), test_size=0.2, random_state=42)
    X_train_BCE_Upgrade, X_test_BCE_Upgrade, y_train_BCE_Upgrade, y_test_BCE_Upgrade = train_test_split(PSE_BCE_t_Upgrade_X, pd.get_dummies(PSE_BCE_t_Upgrade_y), test_size=0.2, random_state=42)
    X_train_TCP_Build, X_test_TCP_Build, y_train_TCP_Build, y_test_TCP_Build = train_test_split(PSE_TCP_t_Build_X, pd.get_dummies(PSE_TCP_t_Build_y), test_size=0.2, random_state=42)
    X_train_TCP_Attack, X_test_TCP_Attack, y_train_TCP_Attack, y_test_TCP_Attack = train_test_split(PSE_TCP_t_Attack_X, pd.get_dummies(PSE_TCP_t_Attack_y), test_size=0.2, random_state=42)

    PSE_BCE_t_SCV_clf_3 = RandomForestClassifier(criterion = 'gini', n_estimators = 100,  max_features = None)
    PSE_BCE_t_t1_clf_3 = RandomForestClassifier(criterion = 'gini', n_estimators = 500,  max_features = None)
    PSE_BCE_t_t2_clf_3 = RandomForestClassifier(criterion = 'gini', n_estimators = 500,  max_features = None)
    PSE_BCE_t_Upgrade_clf_3 = RandomForestClassifier(criterion = 'gini', n_estimators = 500,  max_features = None)
    PSE_TCP_t_Build_clf_3 = RandomForestClassifier(criterion = 'gini', n_estimators = 500,  max_features = None)
    PSE_TCP_t_Attack_clf_3 = RandomForestClassifier(criterion = 'gini', n_estimators = 100,  max_features = None)

    target_ = [y_train_BCE_SCV, y_train_BCE_t1, y_train_BCE_t2, y_train_BCE_Upgrade, y_train_TCP_Build, y_train_TCP_Attack]
    for target in target_:
        print('--------------')
        print(target.shape)
        print('--------------')
        for i in target.keys():
            print(i, sum(target[i]))

    SMOTE_t1_complete_X_train, SMOTE_t1_complete_y_train = SMOTE_(X_train_BCE_t1, y_train_BCE_t1)
    SMOTE_t2_complete_X_train, SMOTE_t2_complete_y_train = SMOTE_(X_train_BCE_t2, y_train_BCE_t2)
    SMOTE_Build_complete_X_train, SMOTE_Build_complete_y_train = SMOTE_(X_train_TCP_Build, y_train_TCP_Build)
    SMOTE_Upgrade_complete_X_train, SMOTE_Upgrade_complete_y_train = SMOTE_(X_train_BCE_Upgrade, y_train_BCE_Upgrade)

    PSE_BCE_t_SCV_clf_3.fit(X_train_BCE_SCV, y_train_BCE_SCV)
    PSE_BCE_t_t1_clf_3.fit(SMOTE_t1_complete_X_train, SMOTE_t1_complete_y_train)
    PSE_BCE_t_t2_clf_3.fit(SMOTE_t2_complete_X_train, SMOTE_t2_complete_y_train)
    PSE_BCE_t_Upgrade_clf_3.fit(SMOTE_Upgrade_complete_X_train, SMOTE_Upgrade_complete_y_train)
    PSE_TCP_t_Build_clf_3.fit(SMOTE_Build_complete_X_train, SMOTE_Build_complete_y_train)
    PSE_TCP_t_Attack_clf_3.fit(X_train_TCP_Attack, y_train_TCP_Attack)

    cm_SCV = confusion_matrix(y_test_BCE_SCV, PSE_BCE_t_SCV_clf_3.predict(X_test_BCE_SCV))
    cm_t1 = confusion_matrix(y_test_BCE_t1.values.argmax(axis=1), PSE_BCE_t_t1_clf_3.predict(X_test_BCE_t1).argmax(axis=1))
    cm_t2 = confusion_matrix(y_test_BCE_t2.values.argmax(axis=1), PSE_BCE_t_t2_clf_3.predict(X_test_BCE_t2).argmax(axis=1))
    cm_Upgrade = confusion_matrix(y_test_BCE_Upgrade.values.argmax(axis=1), PSE_BCE_t_Upgrade_clf_3.predict(X_test_BCE_Upgrade).argmax(axis=1))
    cm_Build = confusion_matrix(y_test_TCP_Build.values.argmax(axis=1), PSE_TCP_t_Build_clf_3.predict(X_test_TCP_Build).argmax(axis=1))
    cm_Attack = confusion_matrix(y_test_TCP_Attack.values.argmax(axis=1), PSE_TCP_t_Attack_clf_3.predict(X_test_TCP_Attack).argmax(axis=1))

    cr_SCV = classification_report(y_test_BCE_SCV, PSE_BCE_t_SCV_clf_3.predict(X_test_BCE_SCV))
    cr_t1 = classification_report(y_test_BCE_t1.values.argmax(axis=1), PSE_BCE_t_t1_clf_3.predict(X_test_BCE_t1).argmax(axis=1))
    cr_t2 = classification_report(y_test_BCE_t2.values.argmax(axis=1), PSE_BCE_t_t2_clf_3.predict(X_test_BCE_t2).argmax(axis=1))
    cr_Upgrade = classification_report(y_test_BCE_Upgrade.values.argmax(axis=1), PSE_BCE_t_Upgrade_clf_3.predict(X_test_BCE_Upgrade).argmax(axis=1))
    cr_Build = classification_report(y_test_TCP_Build.values.argmax(axis=1), PSE_TCP_t_Build_clf_3.predict(X_test_TCP_Build).argmax(axis=1))
    cr_Attack = classification_report(y_test_TCP_Attack.values.argmax(axis=1), PSE_TCP_t_Attack_clf_3.predict(X_test_TCP_Attack).argmax(axis=1))

    print('clf_3 - SCV Accuracy: ',PSE_BCE_t_SCV_clf_3.score(X_test_BCE_SCV, y_test_BCE_SCV))
    print('clf_3 - t1 Accuracy: ',PSE_BCE_t_t1_clf_3.score(X_test_BCE_t1, y_test_BCE_t1))
    print('clf_3 - t2 Accuracy: ',PSE_BCE_t_t2_clf_3.score(X_test_BCE_t2, y_test_BCE_t2))
    print('clf_3 - Upgrade Accuracy: ',PSE_BCE_t_Upgrade_clf_3.score(X_test_BCE_Upgrade, y_test_BCE_Upgrade))
    print('clf_3 - Build Accuracy: ',PSE_TCP_t_Build_clf_3.score(X_test_TCP_Build, y_test_TCP_Build))
    print('clf_3 - Attack Accuracy: ',PSE_TCP_t_Attack_clf_3.score(X_test_TCP_Attack, y_test_TCP_Attack))

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||RadiusNeighborsClassifier
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    X_train_BCE, X_test_BCE, y_train_BCE, y_test_BCE = train_test_split(PSE_BCE_t_X, pd.get_dummies(PSE_BCE_t_y), test_size=0.2, random_state=42)
    X_train_TCP, X_test_TCP, y_train_TCP, y_test_TCP = train_test_split(PSE_TCP_t_X, pd.get_dummies(PSE_TCP_t_y), test_size=0.2, random_state=42)
    X_train_UIE, X_test_UIE, y_train_UIE, y_test_UIE = train_test_split(PSE_UIE_t_X, pd.get_dummies(PSE_UIE_t_y), test_size=0.2, random_state=42)

    y_train_BCE_Unit = y_train_BCE[['BuildHellion', 'BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder',
       'TrainMarine', 'TrainMedivac', 'TrainRaven', 'TrainReaper', 'TrainSCV',
       'TrainViking']]
    y_test_BCE_Unit = y_test_BCE[['BuildHellion', 'BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder',
       'TrainMarine', 'TrainMedivac', 'TrainRaven', 'TrainReaper', 'TrainSCV',
       'TrainViking']]
    y_train_BCE_Upgrade = y_train_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_test_BCE_Upgrade = y_test_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_train_TCP_Build = y_train_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_test_TCP_Build = y_test_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_train_TCP_Attack = y_train_TCP[['Attack']]
    y_test_TCP_Attack = y_test_TCP[['Attack']]

    PSE_BCE_t_Unit_neigh = RadiusNeighborsClassifier(radius = 1)
    PSE_BCE_t_Upgrade_neigh = RadiusNeighborsClassifier(radius = 1)
    PSE_TCP_t_Build_neigh = RadiusNeighborsClassifier(radius = 1)
    PSE_TCP_t_Attack_neigh = RadiusNeighborsClassifier(radius = 1)

    standard_BCE = StandardScaler()
    standard_TCP = StandardScaler()
    standard_UIE = StandardScaler()

    PSE_BCE_t_Unit_neigh.fit(standard_BCE.fit_transform(X_train_BCE), y_train_BCE_Unit)
    PSE_BCE_t_Upgrade_neigh.fit(standard_BCE.fit_transform(X_train_BCE), y_train_BCE_Upgrade)
    PSE_TCP_t_Build_neigh.fit(standard_TCP.fit_transform(X_train_TCP), y_train_TCP_Build)
    PSE_TCP_t_Attack_neigh.fit(standard_TCP.fit_transform(X_train_TCP), y_train_TCP_Attack)

    print('neigh - Unit Accuracy: ',PSE_BCE_t_Unit_neigh.score(standard_BCE.fit_transform(X_train_BCE), y_test_BCE_Unit))
    print('neigh - Upgrade Accuracy: ',PSE_BCE_t_Upgrade_neigh.score(standard_BCE.fit_transform(X_train_BCE), y_test_BCE_Upgrade))
    print('neigh - Build Accuracy: ',PSE_TCP_t_Build_neigh.score(standard_TCP.fit_transform(X_train_TCP), y_test_TCP_Build))
    print('neigh - Attack Accuracy: ',PSE_TCP_t_Attack_neigh.score(standard_TCP.fit_transform(X_train_TCP), y_test_TCP_Attack))

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||KNeighborsClassifier
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

if False:
    X_train_BCE, X_test_BCE, y_train_BCE, y_test_BCE = train_test_split(PSE_BCE_t_X, pd.get_dummies(PSE_BCE_t_y), test_size=0.2, random_state=42)
    X_train_TCP, X_test_TCP, y_train_TCP, y_test_TCP = train_test_split(PSE_TCP_t_X, pd.get_dummies(PSE_TCP_t_y), test_size=0.2, random_state=42)
    X_train_UIE, X_test_UIE, y_train_UIE, y_test_UIE = train_test_split(PSE_UIE_t_X, pd.get_dummies(PSE_UIE_t_y), test_size=0.2, random_state=42)

    y_train_BCE_Unit = y_train_BCE[['BuildHellion', 'BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder',
       'TrainMarine', 'TrainMedivac', 'TrainRaven', 'TrainReaper', 'TrainSCV',
       'TrainViking']]
    y_test_BCE_Unit = y_test_BCE[['BuildHellion', 'BuildSiegeTank', 'BuildThor', 'TrainBanshee',
       'TrainCyclone', 'TrainGhost', 'TrainLiberator', 'TrainMarauder',
       'TrainMarine', 'TrainMedivac', 'TrainRaven', 'TrainReaper', 'TrainSCV',
       'TrainViking']]
    y_train_BCE_Upgrade = y_train_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_test_BCE_Upgrade = y_test_BCE[['UpgradeTerranInfantryWeapons1','UpgradeTerranInfantryArmor1',
        'UpgradeTerranInfantryWeapons2', 'UpgradeTerranInfantryArmor2','UpgradeVehicleWeapons1',
        'UpgradeVehicleWeapons2','ResearchTerranVehicleAndShipArmorsLevel1','ResearchTerranVehicleAndShipArmorsLevel2',
        'ResearchTerranVehicleAndShipArmorsLevel3','UpgradeTerranInfantryWeapons3', 'UpgradeTerranInfantryArmor3',
        'UpgradeVehicleWeapons3','UpgradeShipWeapons2']]
    y_train_TCP_Build = y_train_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_test_TCP_Build = y_test_TCP[['BuildArmory', 'BuildBarracks', 'BuildBarracksReactor','BuildBarracksTechLab',
        'BuildCommandCenter','BuildEngineeringBay', 'BuildFactory', 'BuildFactoryReactor',
        'BuildFactoryTechLab', 'BuildMissileTurret','BuildStarport','BuildStarportReactor',
        'BuildStarportTechLab', 'BuildSupplyDepot']]
    y_train_TCP_Attack = y_train_TCP[['Attack']]
    y_test_TCP_Attack = y_test_TCP[['Attack']]

    PSE_BCE_t_Unit_kneigh = KNeighborsClassifier(weights = 'distance')
    PSE_BCE_t_Upgrade_kneigh = KNeighborsClassifier(weights = 'distance')
    PSE_TCP_t_Build_kneigh = KNeighborsClassifier(weights = 'distance')
    PSE_TCP_t_Attack_kneigh = KNeighborsClassifier(weights = 'distance')

    PSE_BCE_t_Unit_kneigh.fit(X_train_BCE, y_train_BCE_Unit)
    PSE_BCE_t_Upgrade_kneigh.fit(X_train_BCE, y_train_BCE_Upgrade)
    PSE_TCP_t_Build_kneigh.fit(X_train_TCP, y_train_TCP_Build)
    PSE_TCP_t_Attack_kneigh.fit(X_train_TCP, y_train_TCP_Attack)

    print('kneigh - Unit Accuracy: ',PSE_BCE_t_Unit_kneigh.score(X_test_BCE, y_test_BCE_Unit))
    print('kneigh - Upgrade Accuracy: ',PSE_BCE_t_Upgrade_kneigh.score(X_test_BCE, y_test_BCE_Upgrade))
    print('kneigh - Build Accuracy: ',PSE_TCP_t_Build_kneigh.score(X_test_TCP, y_test_TCP_Build))
    print('kneigh - Attack Accuracy: ',PSE_TCP_t_Attack_kneigh.score(X_test_TCP, y_test_TCP_Attack))

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
