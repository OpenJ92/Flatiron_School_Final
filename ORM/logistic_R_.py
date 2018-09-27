#Carry out logistig regression per Map / time. Regress elements along time axis

from routes import *
import matplotlib.pyplot as plt

unique_map_names = [map_['value'] for map_ in get_Game_maps()]
A = event_df_from_Game_TPE_map(unique_map_names[0])

def linear_regression(list_of_feature_names, target_feature, _df, order = 1, norm = True, closed_form = False, steps = 500, show_plot = True, max_lik = True, bias_variance = True):
    samp_df = _df.sample(int(4*(_df.shape[0])/5))
    X_ = order_(order, list_of_feature_names, samp_df)
    y = samp_df[target_feature]

    if norm:
        for i in X_.columns:
            X_[i] = normalize_col(X_[i])[0]
        y = normalize_col(y)[0]

    X_['constant'] = 1

    ini = np.zeros(X_.shape[1])

    for i in np.arange(0,steps):
        err = np.array(y) - (np.array(X_) @ ini)
        grad = np.array(X_).T @ err
        ini = ini + (1/(order*10000))*grad
        print(err @ err)

    X_['y_hat'] = 0
    ##This is not working as expected
    for j in (range(0,len(ini))):
            X_['y_hat'] += ini[j]*X_[X_.columns[j]]

    #print(X_.head())
    if bias_variance:
        print('Model bias measure: ' + str((y - X_['y_hat']).mean()))
        print('Model variance measure: ' + str((X_['y_hat']**2).mean() - (X_['y_hat']).mean()**2))

    if show_plot and len(list_of_feature_names) == 2:
        trace0 = explore_r2(samp_df[list_of_feature_names[0]], samp_df[list_of_feature_names[1]], y, 'red')
        trace1 = explore_r2(samp_df[list_of_feature_names[0]], samp_df[list_of_feature_names[1]], X_['y_hat'], 'blue')
        fig = go.Figure(data=[trace0, trace1])
        offline.plot(fig)
    if show_plot and len(list_of_feature_names) == 3:
        trace0 = explore_r3(samp_df[list_of_feature_names[0]], samp_df[list_of_feature_names[1]], samp_df[list_of_feature_names[2]], y, 'red')
        trace1 = explore_r3(samp_df[list_of_feature_names[0]], samp_df[list_of_feature_names[1]], samp_df[list_of_feature_names[2]], X_['y_hat'], 'blue')
        fig = go.Figure(data=[trace0, trace1])
        offline.plot(fig)

    if max_lik:
        max_likelyhood(X_['y_hat'] - y)

    return ini

def binary_logistic_regression(list_of_feature_names, target_feature, _df, thresh = .5,order = 1, norm = True, steps = 10000, show_tests = True, show_plot = True):
    x = _df
    X = order_(order, list_of_feature_names, x)
    y = _df[target_feature]

    if norm:
        X, col_min_max = normalize_df(X)

    X['Constant'] = 1
    #import pdb; pdb.set_trace()
    theta = np.ones(X.shape[1])
    for step in range(0, steps):
        #print(theta)
        hypothesis = -1 * np.array(X) @ theta
        e_hypothesis = np.e ** hypothesis
        one_e_hypothesis = 1 + e_hypothesis
        inv_one_e_hypothesis = one_e_hypothesis ** -1
        err = inv_one_e_hypothesis - np.array(y)
        grad = np.array(X).T @ err
        theta = theta - (1/10000)*grad

    X['y_hat'] = 0
    X['y_hat'] = -1 * X[X.columns[0:len(X.columns) - 1]] @ theta
    X['y_hat'] = 1 + (np.e ** X['y_hat'])
    X['y_hat'] = X['y_hat'] ** -1

    if show_tests:
        print(binary_classifification_tests(y, X['y_hat'], thresh))

    if show_plot and len(X.columns) == 2:
        plt.plot(x[list_of_feature_names[0]], y, 'blue')
        plt.plot(x[list_of_feature_names[0]], x['y_hat'], 'red')
        plt.show()

    if show_plot and len(x.columns) == 3:
        trace0 = explore_r2(x[list_of_feature_names[0]], x[list_of_feature_names[1]], y, 'blue')
        trace1 = explore_r2(x[list_of_feature_names[0]], x[list_of_feature_names[1]], X['y_hat'], 'red')
        fig = go.Figure(data=[trace0, trace1])
        offline.plot(fig)

    return (X['y_hat'] - y) @ (X['y_hat'] - y)

def order_(order, list_of_feature_names, _df):
    col_names = []
    comb_store = []
    x = 1
    y = ''
    for ord in list(sorted(range(1,order + 1))):
        combs = combinations_with_replacement(list_of_feature_names, ord)
        for comb in combs:
            for element in comb:
                x *= _df[element]
                y += element + ' '
            comb_store.append(x)
            col_names.append(y)
            x = 1
            y = ''
    df_ = pd.concat(comb_store, axis = 1)
    df_.columns = col_names
    return df_

def explore_r2(x,y, _target, color_):
    traces = go.Scatter3d(x = np.array(x).flatten(),
                          y = np.array(y).flatten(),
                          z = _target,
                          mode='markers', marker=dict(size=5, color = color_, opacity=.8))
    return traces

def explore_r3(x,y,z, _target, color_):
        traces = go.Scatter3d(x = np.array(x).flatten(),
                              y = np.array(y).flatten(),
                              z = np.array(z).flatten(),
                              mode='markers', marker=dict(size=5,color = color_, opacity=0))
        return traces

def max_likelyhood(column_, type_ = 'Gauss'):
    mu = sum(column_) / len(column_)
    sig = (np.sqrt(len(column_)) / (np.sqrt(sum(np.multiply((column_ - mu),(column_ - mu))))))**-1
    w = -1*st.norm(loc = mu, scale = sig).logpdf(column_)
    w1 = st.norm(loc = mu, scale = sig).pdf(np.linspace(column_.min(), column_.max(), 1000))
    t = sum(w)
    t1 = np.prod(w1)
    plt.scatter(np.linspace(column_.min(), column_.max(), 1000),w1)
    plt.scatter(column_,np.zeros_like(column_))
    plt.show()
    return (mu, sig)

def normalize_col(_df_col):
    min_max = {'min': _df_col.min(), 'max': _df_col.max()}
    norm = (_df_col - min_max['min']) / (min_max['max'] - min_max['min'])
    return norm, min_max

def normalize_df(_df):
    col_min_max = {}
    for col in _df.columns:
        _df[col], col_min_max[col] = normalize_col(_df[col])
    return _df, col_min_max

def binary_threshold(val, _df, col):
    tf = _df[col] > val
    return tf.astype('int')

def binary_classifification_tests(y, y_hat, val):
    #y, y_hat - dataframe
    X = pd.concat([y,y_hat], axis = 1)
    Y = pd.concat([X, binary_threshold(val, X, 'y_hat')], axis = 1)
    Y.columns = ['y', 'y_hat', 'thresh']

    Y['='] = np.where((Y['thresh'] == Y['y']), Y['y'], np.nan)
    Y['!='] = np.where((Y['thresh'] != Y['y']), Y['y'], np.nan)

    confusion_matrix = {'TP': sum(Y['='] > 0) / len(Y), 'FN': sum(Y['!='] > 0) / len(Y), 'FP': sum(Y['!='] > 0) / len(Y), 'TN': sum(Y['='] == 0) / len(Y)}
    print('Confusion Matrix: ', confusion_matrix)
    print('Precision: ', confusion_matrix['TP'] / (confusion_matrix['TP']+confusion_matrix['FP']))
    print('Recall: ', confusion_matrix['TP'] / sum(Y['y'] == 1))
    print('Accuracy: ', (confusion_matrix['TP'] +  confusion_matrix['TN'])/ (confusion_matrix['TP'] +  confusion_matrix['TN'] + confusion_matrix['FP'] +  confusion_matrix['FN']))
    #import pdb; pdb.set_trace()

    return None

attack_Move = A[A['ability_name'] == 'Attack'][['loc_x', 'loc_y', 'second']]

x_mean = attack_Move['loc_x'].mean()
y_mean = attack_Move['loc_y'].mean()

attack_Move['loc_x'] = attack_Move['loc_x'].apply(lambda x: x - x_mean)
attack_Move['loc_y'] = attack_Move['loc_y'].apply(lambda y: y - y_mean)

A = attack_Move[-1*attack_Move['loc_x'] > attack_Move['loc_y']]
B = attack_Move[-1*attack_Move['loc_x'] <= attack_Move['loc_y']]
A['loc_x'] = -A['loc_x']
A['loc_y'] = -A['loc_y']

attack_Move_concat = pd.concat([A, B])

intervals = 10
attack_Move_concat_interval = [attack_Move_concat[
    (attack_Move_concat['second'] < (attack_Move_concat['second'].max() / intervals)* (interval - intervals) + attack_Move_concat['second'].max())
    &
    (attack_Move_concat['second'] > (attack_Move_concat['second'].max() / intervals)* ((interval - 1)- intervals) + attack_Move_concat['second'].max())]
    for interval in range(1, intervals))]


#binary_logistic_regression(['loc_x', 'loc_y'], target_feature, _df, thresh = .5,order = 1, norm = True, steps = 10000, show_tests = True, show_plot = True)
#####Thnk about how one might rotate and translate points
