import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from __init__ import app
##Change this to the furthest file on branch.
from routes import *

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#app Layout|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

app.layout = html.Div(children = [
    dcc.Tabs(id = 'tabs', value = 1, children = [
        dcc.Tab(label = 'PSE by Player', value = 1),
        #dcc.Tab(label = 'UBE/PSE by Game', value = 2),
        dcc.Tab(label = 'Events Aggregate', value = 3),
        dcc.Tab(label = 'Events PCA', value = 4),
        #dcc.Tab(label = 'Events PCA Unsupervised', value = 5)
    ]),
    html.Div(id = 'tab_Output')
])

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#call-backs|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

@app.callback(Output(component_id = 'tab_Output', component_property = 'children'),
                [Input(component_id = 'tabs', component_property = 'value')])
def display_content(value):
    if value == 1:
        return [
            html.H3(children='PSE by Player'),
            dcc.Dropdown(id = 'PlayerStatsEventPlayer-options', options = get_PSE_label()),
            dcc.Dropdown(id = 'Player-options', options = get_Player_names()),
            dcc.Graph(id='PlayerStatsEventPlayer-graph')
                ]
    elif value == 2:
        return [
            html.H3(children='UBE/PSE by Game'),
            dcc.Dropdown(id = 'Game-options_3', options = [{'label': 'PSE', 'value': 1}, {'label': 'UBE', 'value': 2}]),
            dcc.Dropdown(id = 'Game-options_2', options = get_Game_names()),
            dcc.Graph(id='PlayerStatsEvent-graph_2')
                ]
    elif value == 3:
        return [
            html.H3(children = 'Events Aggregate'),
            dcc.Dropdown(id = 'Game-options_4', options = [{'label': 'Terran', 'value': 'Terran'}, {'label': 'Zerg', 'value': 'Zerg'}, {'label': 'Protoss', 'value': 'Protoss'}]),
            dcc.Dropdown(id = 'Game-options_4_1', options = [{'label': 'BCE', 'value': 'BCE'}, {'label': 'TPE', 'value': 'TPE'},
                                                             {'label': 'UBE', 'value': 'UBE'}, {'label': 'UDE', 'value': 'UDE'},
                                                             {'label': 'PSE', 'value': 'PSE'}]),
            html.Iframe(id = 'plotly_agg_out',srcDoc = open('plotly_files/Terran_BCE_Agg.html', 'r').read(), width ='100%', height= '900')
        ]
    elif value == 4:
        return [
            html.H3(children = 'Events PCA'),
            dcc.Dropdown(id = 'Game-options_5', options = [{'label': 'Terran', 'value': 'Terran'}, {'label': 'Zerg', 'value': 'Zerg'}, {'label': 'Protoss', 'value': 'Protoss'}]),
            dcc.Dropdown(id = 'Game-options_5_1', options = [{'label': 'BCE', 'value': 'BCE'}, {'label': 'TPE', 'value': 'TPE'},
                                                             {'label': 'UBE', 'value': 'UBE'}, {'label': 'UDE', 'value': 'UDE'},
                                                             {'label': 'PSE', 'value': 'PSE'}]),
            html.Iframe(id = 'plotly_PCA_out',srcDoc = open('plotly_files/Terran_BCE_PCA.html', 'r').read(), width ='100%', height= '900')
        ]
    elif value == 5:
        return [
            html.H3(children = 'Events PCA Cluster'),
            dcc.Dropdown(id = 'race_6', options = [{'label': 'Terran', 'value': 'Terran'}, {'label': 'Zerg', 'value': 'Zerg'}, {'label': 'Protoss', 'value': 'Protoss'}]),
            dcc.Dropdown(id = 'event_6', options = [{'label': 'BCE', 'value': 'BCE'}, {'label': 'TPE', 'value': 'TPE'},
                                                             {'label': 'UBE', 'value': 'UBE'}, {'label': 'UDE', 'value': 'UDE'},
                                                             {'label': 'PSE', 'value': 'PSE'}]),
            dcc.Dropdown(id = 'unsupervised_6', options = [{'label': 'GaussianMixture', 'value': 'GM'}, {'label': 'KMeans', 'value': 'KM'}]),
            dcc.Dropdown(id = 'n_clusters_6', options = [{'label': str(i), 'value': str(i)} for i in range(2,10)]),
            html.Iframe(id = 'plotly_PCA_unsupervised_out',srcDoc = open('plotly_files/Protoss_BCE_GM_2_unsupervised.html', 'r').read(), width ='100%', height= '900')
        ]


@app.callback(Output(component_id = 'PlayerStatsEventPlayer-graph', component_property = 'figure'),
                [Input(component_id = 'PlayerStatsEventPlayer-options', component_property = 'value'),
                Input(component_id = 'Player-options', component_property = 'value')])
def construct_PSEGraph_Player(input_1, input_2):
    player_ = filter_by_Player_name(input_2)
    player_PSE = player_.events_PSE
    player_PSE_df = pd.DataFrame([vars(event) for event in player_PSE])

    return {'data' : [go.Scatter(x = player_PSE_df[player_PSE_df['player_id'] == player_.id]['second'], y = player_PSE_df[player_PSE_df['player_id'] == player_.id][input_1], mode = 'markers')],
            'layout' : go.Layout(title= "PSE - " + input_1 + ' - ' + input_2, height = 400, width = 700)}

@app.callback(Output(component_id = 'PlayerStatsEvent-graph_2', component_property = 'figure'),
                [Input(component_id = 'Game-options_2', component_property = 'value'),
                    Input(component_id = 'Game-options_3', component_property = 'value')])
def construct_PSEHeatmep_Game(input_1, input_2):
    game_ = filter_Game_by_name(input_1)
    if input_2 == 1:
        game_PSE = game_.events_PSE
        filtered_game_PSE = pd.DataFrame([vars(event) for event in game_PSE])
        df = pd.DataFrame(filtered_game_PSE).drop(columns = ['_sa_instance_state', 'id', 'game_id', 'name'])
        df = df.sort_values(by = ['player_id', 'second'])
        df = df.drop(columns = ['player_id', 'second'])
        filtered_game_PSE_ = np.array(df).T
        return {'data' : [go.Heatmap(z = filtered_game_PSE_, y = df.columns)],
                'layout' : go.Layout(title= "PSE - " + input_1, height = 500, width = 1000)}
    if input_2 == 2:
        game_PSE = game_.events_UBE
        filtered_game_PSE = pd.DataFrame([vars(event) for event in game_PSE])
        df = pd.DataFrame(filtered_game_PSE).drop(columns = ['_sa_instance_state', 'id', 'game_id', 'name', 'loc_x', 'loc_y'])
        df = df.sort_values(by = ['player_id', 'second'])
        df = pd.get_dummies(df).drop(columns = ['unit_type_name_BeaconArmy', 'unit_type_name_BeaconAttack',  'unit_type_name_BeaconAuto',
                                                'unit_type_name_BeaconClaim', 'unit_type_name_BeaconCustom1', 'unit_type_name_BeaconCustom2',
                                                'unit_type_name_BeaconCustom3', 'unit_type_name_BeaconCustom4', 'unit_type_name_BeaconDefend',
                                                'unit_type_name_BeaconDetect', 'unit_type_name_BeaconExpand', 'unit_type_name_BeaconHarass',
                                                'unit_type_name_BeaconIdle', 'unit_type_name_BeaconRally', 'unit_type_name_BeaconScout'])
        df_A = df[df['player_id'] == df['player_id'].unique()[0]].cumsum()
        df_B = df[df['player_id'] == df['player_id'].unique()[1]].cumsum()
        df = pd.concat([df_A, df_B], axis = 0)
        df = df.drop(columns = ['player_id', 'second'])
        filtered_game_PSE_ = np.array(df).T
        return {'data' : [go.Heatmap(z = filtered_game_PSE_, y = df.columns)],
                'layout' : go.Layout(title= "PSE - " + input_1, height = 500, width = 1000)}

@app.callback(Output(component_id = 'plotly_agg_out', component_property = 'srcDoc'),
                [Input(component_id = 'Game-options_4', component_property = 'value'),
                    Input(component_id = 'Game-options_4_1', component_property = 'value')])
def load_plotly_agg(input_1, input_2):
    #print('/Users/flatironschool/Desktop/PySC2/sc2reader/ORM/plotly_files/' + input_1 + '_' + input_2 + '_Agg.html')
    return open('/Users/flatironschool/Desktop/PySC2/sc2reader/ORM/plotly_files/' + input_1 + '_' + input_2 + '_Agg.html', 'r').read()

@app.callback(Output(component_id = 'plotly_PCA_out', component_property = 'srcDoc'),
                [Input(component_id = 'Game-options_5', component_property = 'value'),
                    Input(component_id = 'Game-options_5_1', component_property = 'value')])
def load_plotly_PCA(input_1, input_2):
    #print('/Users/flatironschool/Desktop/PySC2/sc2reader/ORM/plotly_files/' + input_1 + '_' + input_2 + '_Agg.html')
    return open('/Users/flatironschool/Desktop/PySC2/sc2reader/ORM/plotly_files/' + input_1 + '_' + input_2 + '_PCA.html', 'r').read()

@app.callback(Output(component_id = 'plotly_PCA_unsupervised_out', component_property = 'srcDoc'),
                [Input(component_id = 'race_6', component_property = 'value'),
                    Input(component_id = 'event_6', component_property = 'value'),
                        Input(component_id = 'unsupervised_6', component_property = 'value'),
                            Input(component_id = 'n_clusters_6', component_property = 'value')])
def load_plotly_PCA(input_1, input_2, input_3, input_4):
    #print('/Users/flatironschool/Desktop/PySC2/sc2reader/ORM/plotly_files/' + input_1 + '_' + input_2 + '_Agg.html')
    return open('/Users/flatironschool/Desktop/PySC2/sc2reader/ORM/plotly_files/' + input_1 + '_' + input_2 + '_' + input_3 + '_' + input_4 +'_unsupervised.html', 'r').read()

###Add:
#----- PCA_df plot
#----- PCA_FPC plot
#----- Cluster plot
#----- attempt at displaying cluster content via explore (All Professional Replays)


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#run||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


if __name__ == '__main__':
    app.run_server(debug=True)
