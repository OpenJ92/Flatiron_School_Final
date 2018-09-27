import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from __init__ import app
from routes import *

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#app Layout|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

app.layout = html.Div(children = [
    dcc.Tabs(id = 'tabs', value = 1, children = [
        dcc.Tab(label = 'PSE by Player', value = 1),
        dcc.Tab(label = 'UBE/PSE by Game', value = 2)
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
            dcc.Dropdown(id = 'PlayerStatsEventPlayer-options', options = get_PSE_label(), value = get_PSE_label()[1]),
            dcc.Dropdown(id = 'Player-options', options = get_Player_names(), value = get_Player_names()[0]),
            dcc.Graph(id='PlayerStatsEventPlayer-graph')
                ]
    elif value == 2:
        return [
            html.H3(children='UBE/PSE by Game'),
            dcc.Dropdown(id = 'Game-options_3', options = [{'label': 'PSE', 'value': 1}, {'label': 'UBE', 'value': 2}], value = 1),
            dcc.Dropdown(id = 'Game-options_2', options = get_Game_names(), value = get_Game_names()[0]),
            dcc.Graph(id='PlayerStatsEvent-graph_2')
                ]

@app.callback(Output(component_id = 'PlayerStatsEventPlayer-graph', component_property = 'figure'),
                [Input(component_id = 'PlayerStatsEventPlayer-options', component_property = 'value'),
                Input(component_id = 'Player-options', component_property = 'value')])
def construct_PSEGraph_Player(input_1, input_2):
    player_ = filter_by_Player_name(input_2)
    player_PSE = player_.events_PSE
    player_PSE_df = pd.DataFrame([vars(event) for event in player_PSE])

    return {'data' : [go.Scatter(x = player_PSE_df[player_PSE_df['player_id'] == player_.id]['second'], y = player_PSE_df[player_PSE_df['player_id'] == player_.id][input_1], mode = 'markers')],
            'layout' : go.Layout(title= "PSE - " + input_1 + ' - ' + input_2, height = 700, width = 1400)}

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
                'layout' : go.Layout(title= "PSE - " + input_1, height = 700, width = 1400)}
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
                'layout' : go.Layout(title= "PSE - " + input_1, height = 700, width = 1400)}
    if input_2 == 3:
        #BCE
        pass

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#run||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


if __name__ == '__main__':
    app.run_server(debug=True)
