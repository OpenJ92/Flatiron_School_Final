import numpy as np
import sklearn
import os
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go
from plotly import tools
from PCA_ETL import construct_df_UnitsStructures, construct_full_UnitsStructures_df, combine_df_UnitStructures
offline.init_notebook_mode()

## Once PCA elements are complete, look to cluster along each of the above imported elements.
## Load functions:
##      1. Load by specific parameters. (Race, UserID, ParticipantID, GameID)
## Fit:
##      1. from nltk.cluster.kmeans import KMeansClusterer -- Use Cosine Similarity measure on this unsupervised event.
##      2. from sklearn.mixture import GaussianMixture
##      3. Evaluate above models: from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples
##          for Aggregate true, use radial RSS as a means to measure goodness of fit.
##      4. Determine the composition of each cluster with respect to the elements clustered on.


# Consider how one can filter on n parameters. Should it be done here or in the Database query?
def load_decomposition(obj, time, aggregate, name_decomp, name_normalization, event_name):
    if isinstance(obj, Participant):
        load_Participant_decomposition(obj, time, aggregate, name_decomp, name_normalization, event_name)
        # Return the participant decomposition of this type. In load_Participant_decomposition, check to see if that participant's id is in that directory's csv.
        # example: [participant_Decomposition]
        pass
    elif isinstance(obj, Game):
        load_Game_decomposition(obj,time, aggregate, name_decomp, name_normalization, event_name)
        # Return the participants decomposition belonging to game object
        # example: [participant_1_Decompostion, particiapnt_2_Decompostion]
        pass
    elif isinstance(obj, User):
        load_User_decomposition(obj,time, aggregate, name_decomp, name_normalization, event_name)
        # Return all participant decompositions assosiated with the passed in User id.
        # example: [participant_decompostionA, participant_decompositionB, ....., participant_decompositionN, ...]
        pass
    elif obj in ['Protoss', 'Terran', 'Zerg']:
        load_Playrace_decomposition(obj,time, aggregate, name_decomp, name_normalization, event_name)
        # Return all participant decompositions assosiated with particular playrace.
        # example: [participant_decompostionA, participant_decompositionB, ....., participant_decompositionN, ...]
    elif isinstance(obj, int):
        load_League_decomposition(obj,time, aggregate, name_decomp, name_normalization, event_name)
        # Return all participant deccompositions assosiated with particular league
        # example: [participant_decompostionA, participant_decompositionB, ....., participant_decompositionN, ...]
    else:
        print('Error: Unsupported type.')

def load_Decomposition(participant, time, aggragate, name_decomp, name_normalization, event_name):
    decomposition_load = joblib.load(event_name + '_' + name_decomp + '_' + name_normalization + '_time' + str(time) + '_agg' + str(aggragate) + '_Models/' + event_name + '_' +
    str(participant.id) + '_' + str(participant.user[0].id) + '_' + participant.playrace + '_' +
    str(participant.game[0].id) + '_' + participant.league +'.joblib')
    return decomposition_load

def load_Decomposition_FSV(participant, time, aggragate, name_decomp, name_normalization, event_name):
    load_Decomp = load_Decomposition(participant, time, aggragate, name_decomp, name_normalization, event_name).components_[0]
    load_Decomp = -1 * load_Decomp if (load_Decomp @ np.ones_like(load_Decomp) < 0) else load_Decomp
    return load_Decomp

def load_Decomposiation_batch(sql_func, name_decomp, name_normalization, event_name):
    participants = sql_func[:300]
    return [load_Decomposition(participant, name_decomp, name_normalization, event_name) for participant in participants]

def load_Decomposition_batch_FSV(sql_func, name_decomp, name_normalization, event_name):
    participants = sql_func[:500]
    singular_vector_decomposition = [pd.DataFrame(load_Decomposition_FSV(participant, time, aggragate, name_decomp, name_normalization, event_name)) for participant in participants]
    singular_vector_decomposition_DataFrame = pd.concat(singular_vector_decomposition, axis = 1, sort = False).T
    return singular_vector_decomposition_DataFrame

def radial_RSS(participant, time, aggragate, name_decomp, name_normalization, event_name):
    singular_vector = load_Decomposition_FSV(participant, time, aggragate, name_decomp, name_normalization, event_name)
    singular_vector = -1*singular_vector if ((singular_vector @ np.ones_like(singular_vector)) < 0) else singular_vector
    event_df = construct_full_UnitsStructures_df(participant, event_name, time, aggragate).drop(columns = ['second', 'participant_id'])
    min_max = MinMaxScaler()
    event_df_mm = pd.DataFrame(min_max.fit_transform(event_df))
    inner_product = np.arccos((event_df_mm @ singular_vector) * (1 / event_df_mm.apply(np.linalg.norm, axis = 1))) * event_df_mm.apply(np.linalg.norm, axis = 1)
    return event_df_mm.apply(np.linalg.norm, axis = 1), inner_product

def cummalative_radial_RSS(participant, name_decomp, name_normalization, event_name):
    # look at this function closer. The elements should be ordered by np.linalg.norm then cumal sum
    inner_product = radial_RSS(participant, name_decomp, name_normalization, event_name)[1]
    inner_product_ = inner_product * inner_product
    inner_product_cummalative = inner_product_.cumsum()
    return inner_product_cummalative

def full_radial_RSS(participant, name_decomp, name_normalization, event_name):
    inner_product = radial_RSS(participant, name_decomp, name_normalization, event_name)
    return (inner_product[1]) @ (inner_product[1]).T

def plot_radial_RSS(participant, name_decomp, name_normalization, event_name):
    # for use with radial_RSS and cummalative_radial_RSS
    X, y = radial_RSS(participant, name_decomp, name_normalization, event_name)
    plt.scatter(X,y)
    plt.show()

def plot_(DataFrame, name):
    #For use with load_Decomposition_batch_FSV,
    #             construct/combine_full_UnitsStructures_df,
    #             construct/combine_df_UnitsStructures
    DataFrame = DataFrame.drop(columns = [q for q in ['second', 'participant_id', 'participant_id_x', 'participant_id_y'] if q in DataFrame.columns])
    principle_component_analysis = PCA(n_components = 3)
    min_max = MinMaxScaler()
    DataFrame = min_max.fit_transform(DataFrame)
    X = principle_component_analysis.fit_transform(DataFrame)
    explore_r3(X[:,0], X[:,1], X[:,2], name)
    #look to integrate plot funcion with DASH app

def plot_shell_df(participant, name_decomp, name_normalization, event_name, name, plot = True):
    FSV = load_Decomposition_FSV(participant, name_decomp, name_normalization, event_name)
    FSV = FSV if (FSV @ np.ones_like(FSV) > 0) else -1*FSV
    FSV_df = pd.DataFrame([i*FSV for i in np.linspace(0,2,500)], columns = unique_event_names()[event_name])
    UnitStructures = construct_full_UnitsStructures_df(participant, event_name, time = True).drop(columns = ['second', 'participant_id'])
    min_max = MinMaxScaler()
    UnitStructures_ = pd.DataFrame(min_max.fit_transform(UnitStructures), columns = unique_event_names()[event_name])
    FSV_UnitStructures_df = pd.concat([UnitStructures_, FSV_df], sort = False)
    if plot:
        principle_component_analysis = PCA(n_components = 3)
        X = principle_component_analysis.fit_transform(FSV_UnitStructures_df)
        explore_r3(X[:,0], X[:,1], X[:,2], name)
    return FSV_UnitStructures_df

def multiplot_shell_df(sqlfunc, name_decomp, name_normalization, event_name, name):
    participants = sqlfunc[30:50]
    collect_data = [plot_shell_df(participant, name_decomp, name_normalization, event_name, name, plot = False) for participant in participants]
    concat_data = pd.concat(collect_data, sort = False)
    principle_component_analysis = PCA(n_components = 3)
    projected_data = principle_component_analysis.fit_transform(concat_data)
    traces = [go.Scatter3d(x = projected_data[:,0],y = projected_data[:,1],z = projected_data[:,2], mode='markers', marker = dict(size=3))]
    fig = go.Figure(data=traces)
    offline.plot(fig, filename = 'plotly_files/' + name + '.html', auto_open=True)

def multiplot_Singular_Vectors(sql_func, name_decomp, name_normalization, event_name, name):
    singular_vectors = load_Decomposition_batch_FSV(sql_func, name_decomp, name_normalization, event_name)
    pca = PCA(n_components = 3)
    X = pca.fit_transform(singular_vectors)
    import pdb; pdb.set_trace()
    explore_r3(X[:,0], X[:,1], X[:,2], name)
