def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import timeit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import copy
import json
from time import strftime, localtime

pstart = timeit.default_timer()
print("=========")
print('Start: {}'.format(strftime('%c', localtime())))
print("=========")

def remove_hyperparam(hyperparam, paramdict):
    paramcopy = copy(paramdict)
    paramcopy.pop(hyperparam)
    return json.dumps(paramcopy)

def paramstr(paramdict):
    return json.dumps(paramdict)

def runcv(label, learner, params, X_train, y_train):
    learnercv = GridSearchCV(learner, params, scoring=make_scorer(accuracy_score))
    learnercv.fit(X_train, y_train)    
    results = {
        'best_score': learnercv.best_score_,
        'best_params': learnercv.best_params_,
        'best_estimator': learnercv.best_estimator_,
        'mean_fit_time': learnercv.cv_results_['mean_fit_time'][learnercv.best_index_],
        'mean_score_time': learnercv.cv_results_['mean_score_time'][learnercv.best_index_],
        'n_training': X_train.shape[0],
        'cv_results': learnercv.cv_results_ 
    }
    #print('{} done.'.format(label))
    #print('{} CV best accuracy score: {}'.format(label, learnercv.best_score_))
    #print('{} CV best params: {}'.format(label, learnercv.best_params_))
    return results

def plot_learning_curves_with_time(label, lcdata, paramidx):
    fig, ax1 = plt.subplots()
    ax1.fill_between(lcdata[label]['n'], lcdata[label]['mean_train_score'] - lcdata[label]['std_train_score'],
                     lcdata[label]['mean_train_score'] + lcdata[label]['std_train_score'], alpha=0.1,)
    ax1.fill_between(lcdata[label]['n'], lcdata[label]['mean_test_score'] - lcdata[label]['std_test_score'],
                     lcdata[label]['mean_test_score'] + lcdata[label]['std_test_score'], alpha=0.1)
    pts1 = ax1.plot(lcdata[label]['n'], lcdata[label]['mean_train_score'], 'o-',
                 label="Training")
    pts2 = ax1.plot(lcdata[label]['n'], lcdata[label]['mean_test_score'], 'o-',
                 label="Validation")
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Accuracy Score')
    ax2 = ax1.twinx()
    pts3 = ax2.plot(lcdata[label]['n'], lcdata[label]['mean_fit_time'], '.-',
                     label="Fit Time", linestyle='dashed', color="green", alpha=0.5)
    pts4 = ax2.plot(lcdata[label]['n'], lcdata[label]['mean_score_time'], '.-',
                     label="Score Time", linestyle='dashed', color="gray", alpha=0.5)
    pts = pts1 + pts2 + pts3 + pts4
    labs = [p.get_label() for p in pts]
    ax2.set_ylabel('Time (secs)', color="green")
    ax2.tick_params(axis='y', labelcolor='green')
    plt.suptitle('{} Learning Curve'.format(label), y=1.02)
    plt.title(results[label][0]['cv_results']['params'][paramidx], fontsize=10)
    plt.legend(pts, labs, loc='best')
    fig.tight_layout()
    fig = plt.gcf()
    fig.savefig(''.join(['lc-sign-lang-',label,'-',str(paramidx),'.png']), bbox_inches="tight")
    plt.close()

def plot_model_complexity_with_time(label, hyperparam, mcdata, paramtext, dataidx, pisuffix, ignore_ticks=False):
    fig, ax1 = plt.subplots()
    ax1.fill_between(mcdata[label][hyperparam], mcdata[label]['mean_train_score'] - mcdata[label]['std_train_score'],
                     mcdata[label]['mean_train_score'] + mcdata[label]['std_train_score'], alpha=0.1)
    ax1.fill_between(mcdata[label][hyperparam], mcdata[label]['mean_test_score'] - mcdata[label]['std_test_score'],
                     mcdata[label]['mean_test_score'] + mcdata[label]['std_test_score'], alpha=0.1)
    pts1 = ax1.plot(mcdata[label][hyperparam], mcdata[label]['mean_train_score'], 'o-',
                 label="Training")
    pts2 = ax1.plot(mcdata[label][hyperparam], mcdata[label]['mean_test_score'], 'o-',
                 label="Validation")
    if not ignore_ticks:
        ax1.set_xticks(mcdata[label][hyperparam])
    if label == 'SVM' and hyperparam == 'C':
        ax1.set_xlabel('log10(C)')
    else:
        ax1.set_xlabel(hyperparam)
    ax1.set_ylabel('Accuracy Score')
    ax2 = ax1.twinx()
    pts3 = ax2.plot(mcdata[label][hyperparam], mcdata[label]['mean_fit_time'], '.-',
                     label="Fit Time", linestyle='dashed', color="green", alpha=0.5)
    pts4 = ax2.plot(mcdata[label][hyperparam], mcdata[label]['mean_score_time'], '.-',
                     label="Score Time", linestyle='dashed', color="gray", alpha=0.5)
    pts = pts1 + pts2 + pts3 + pts4
    labs = [p.get_label() for p in pts]
    ax2.set_ylabel('Time (secs)', color="green")
    ax2.tick_params(axis='y', labelcolor='green')
    plt.suptitle('{} Model Complexity'.format(label), y=1.02)
    plt.title(paramtext, fontsize=10)
    plt.legend(pts, labs, loc='best')
    fig = plt.gcf()    
    fig.tight_layout()
    fig.savefig(''.join(['mc-sign-lang-',label,'-',str(dataidx),'-',pisuffix,'.png']), bbox_inches="tight")
    plt.close()

def plot_valid_vs_test(xlabels, vals, legends, ylim, ylabel, figname):
    width = 0.8
    n = len(vals)
    _X = np.arange(len(xlabels))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge")   
        plt.xticks(_X, xlabels)
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel(ylabel)
    plt.legend(legends)
    plt.savefig(figname)
    plt.close()
    
def setup_plot_2bar(data1, data2, legends, ylim, ylabel, filename):
  vals = []
  xlabels = sorted(data1.keys())
  val1 = []
  for xl in xlabels:
      val1.append(data1[xl])
  val2 = []
  for xl in xlabels:
      val2.append(data2[xl])
  vals = [val1, val2]
  if ylim is None:
    ylim = [min([min(val1), min(val2)])-0.5, max([max(val1), max(val2)])+0.5]
  plot_valid_vs_test(xlabels, vals, legends, ylim, ylabel, filename)

print
print('Reading file...')
data = pd.read_csv('sign_mnist_train.csv')

percentages = map(lambda x: x*100./len(data['label']), data['label'].value_counts().sort_index())
print('labels min percentage: {}'.format(min(percentages)))
print('labels max percentage: {}'.format(max(percentages)))

seed = 0
feature_columns = data.columns[1:]
print(feature_columns)
print('feature columns count: {}'.format(len(feature_columns)))

print
print('Splitting data for each dataset size...')
data_sizes = list((np.arange(4,9,0.5)*1000).astype(int))
print('data sizes: ', data_sizes)

splitdata = []
for n in data_sizes:
   ndata = data.sample(n=int(n), random_state=seed)
   X = ndata.loc[:, feature_columns]
   y = ndata.loc[:, 'label']
   scaler = MinMaxScaler(feature_range=(0,1))
   X_scaled = scaler.fit_transform(X)
   splitdata.append({
       'X_train': X_scaled,
       'y_train': y
   })

print
print('Setting parameters...')
params = {}
params['DT'] = {'max_depth': [5, 10, 15, 20, 25, 30, 50, 100],
               'max_features': [0.2, 1.0]}
params['KNN'] = {'n_neighbors': [1, 3, 5, 7, 10, 12, 15, 20, 30, 40, 50],
                 'weights': ['distance']}
params['SVM'] = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
                 'gamma': [0.001], 
                 'kernel': ['linear']}
params['NN'] = {'activation': ['tanh'], 
                'hidden_layer_sizes': [1,3,5,10,25,50,75,100,125,150,175],
                'alpha': [0.1]
               }
params['AB'] = {'learning_rate': [0.1],
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]}


classifiers = ['DT', 'KNN', 'AB', 'SVM', 'NN']
results = {}
for classifier in classifiers:
    results[classifier] = []

print
print('Training models...')
for s in splitdata:
    X_train = s['X_train']
    y_train = s['y_train']
    print('X_train size: {} ...'.format(X_train.shape[0]))
    cvout = runcv('DT', DecisionTreeClassifier(random_state=seed), params['DT'], X_train, y_train)
    results['DT'].append(cvout)
    cvout = runcv('KNN', KNeighborsClassifier(), params['KNN'], X_train, y_train)
    results['KNN'].append(cvout)
    cvout = runcv('AB', AdaBoostClassifier(random_state=seed), params['AB'], X_train, y_train)
    results['AB'].append(cvout)
    cvout = runcv('SVM', SVC(random_state=seed), params['SVM'], X_train, y_train)
    results['SVM'].append(cvout)
    cvout = runcv('NN', MLPClassifier(random_state=seed, max_iter=1000), params['NN'], X_train, y_train)
    results['NN'].append(cvout)

#### lcurve
print
print('Plotting learning curves...')

useindex = {}
useindex['DT'] = 0
useindex['KNN'] = 0
useindex['AB'] = 0
useindex['SVM'] = 0
useindex['NN'] = 0

n_datasets = len(data_sizes)
for classifier in classifiers:
    cvparams = results[classifier][0]['cv_results']['params']
    for i in range(len(results[classifier][0]['cv_results']['params'])):
        if (classifier == 'DT' and 
            (cvparams[i]['max_depth'] != 30 or
             cvparams[i]['max_features'] not in [0.2, 1.0])):
            continue
        if (classifier == 'AB' and 
            cvparams[i]['n_estimators'] != 300):
            continue
        if (classifier == 'KNN' and 
            cvparams[i]['n_neighbors'] != 1):
            continue
        if (classifier == 'NN' and
            cvparams[i]['hidden_layer_sizes'] != 100):
            continue
        if (classifier == 'SVM' and
            cvparams[i]['C'] != 1
           ):
            continue
        learning_curves = {}
        lcurve_x = []
        lcurve_y_tr = []
        lcurve_y_ts = []
        lcurve_y_std_tr = []
        lcurve_y_std_ts = []
        lcurve_y_score_tm = []
        lcurve_y_fit_tm = []
        useindex[classifier] = i
        for n in range(n_datasets):
            nsizeout = results[classifier][n]
            cvresults = nsizeout['cv_results']
            lcurve_x.append(nsizeout['n_training'])
            lcurve_y_tr.append(cvresults['mean_train_score'][useindex[classifier]])
            lcurve_y_ts.append(cvresults['mean_test_score'][useindex[classifier]])
            lcurve_y_std_tr.append(cvresults['std_train_score'][useindex[classifier]])
            lcurve_y_std_ts.append(cvresults['std_test_score'][useindex[classifier]])
            lcurve_y_score_tm.append(cvresults['mean_score_time'][useindex[classifier]])
            lcurve_y_fit_tm.append(cvresults['mean_fit_time'][useindex[classifier]])
        learning_curves[classifier] = (
            pd.DataFrame(np.vstack((lcurve_x, lcurve_y_tr, lcurve_y_ts, lcurve_y_std_tr, lcurve_y_std_ts, lcurve_y_score_tm, lcurve_y_fit_tm)).T, 
                         columns=['n', 'mean_train_score', 'mean_test_score', 'std_train_score', 'std_test_score', 'mean_score_time', 'mean_fit_time']))
        plot_learning_curves_with_time(classifier, learning_curves, i)

#### mcurve
print
print('Plotting model complexity curves...')
import math
hyperparams = {}
hyperparams['NN'] = 'hidden_layer_sizes'
hyperparams['SVM'] = 'C'
hyperparams['AB'] = 'n_estimators'
hyperparams['KNN'] = 'n_neighbors'
hyperparams['DT'] = 'max_depth'
ignore_ticks = False
for classifier in classifiers:
    if (classifier == 'SVM'):
        ignore_ticks = True
    hyperparam = hyperparams[classifier]
    for dataidx in [9]:
        ciresults = results[classifier][dataidx]
        cvresults = ciresults['cv_results']
        n_params = len(cvresults['params'])
        uparams = []
        for pi in range(n_params):
            uparams.append(remove_hyperparam(hyperparam, cvresults['params'][pi]))
        uparams = list(set(uparams))
        for u in uparams:
            mcurve_x = []
            mcurve_y_tr = []
            mcurve_y_ts = []
            mcurve_y_score_tm = []
            mcurve_y_fit_tm = []
            mcurve_y_std_tr = []
            mcurve_y_std_ts = []
            model_complexity_data = {}
            pis = []
            for pi in range(n_params):
                rparams = cvresults['params'][pi]
                if (classifier == 'NN' and
                    (rparams['learning_rate'] == 'invscaling' or
                     rparams['solver'] == 'sgd')):
                    continue
                if (classifier == 'SVM' and
                    rparams['kernel'] == 'sigmoid'
                   ):
                    continue
                rstr = remove_hyperparam(hyperparam, rparams)
                if u != rstr:
                    continue
                if (rparams[hyperparam] == None):
                    continue
                if classifier == 'SVM' and hyperparams[classifier] == 'C':
                    mcurve_x.append(math.log10(rparams[hyperparam]))
                else:
                    mcurve_x.append(rparams[hyperparam])
                pis.append(str(pi))
                mcurve_y_tr.append(cvresults['mean_train_score'][pi])
                mcurve_y_ts.append(cvresults['mean_test_score'][pi])
                mcurve_y_score_tm.append(cvresults['mean_score_time'][pi])
                mcurve_y_fit_tm.append(cvresults['mean_fit_time'][pi])
                mcurve_y_std_tr.append(cvresults['std_train_score'][pi])
                mcurve_y_std_ts.append(cvresults['std_test_score'][pi])
            if len(mcurve_y_tr) == 0:
                continue
            model_complexity_data[classifier] = pd.DataFrame(
                np.vstack((mcurve_x, mcurve_y_tr, mcurve_y_ts, mcurve_y_std_tr, mcurve_y_std_ts, mcurve_y_score_tm, mcurve_y_fit_tm)).T, 
                columns=[hyperparam, 'mean_train_score', 'mean_test_score', 'std_train_score', 'std_test_score', 'mean_score_time', 'mean_fit_time'])
            if u == '{}':
                u = ''
            pisuffix = '-'.join(pis)
            paramtext = "n = {}, {}".format(ciresults['n_training'], u)
            plot_model_complexity_with_time(classifier, hyperparam, model_complexity_data, paramtext, dataidx, pisuffix, ignore_ticks)

#### validation vs test
validation_scores = {}
for classifier in classifiers:
    mean_test_scores = results[classifier][9]['cv_results']['mean_test_score']
    validation_scores[classifier] = max(mean_test_scores)
    for p in range(len(mean_test_scores)):
      if validation_scores[classifier] == mean_test_scores[p]:
         print('best param: {}'.format(results[classifier][9]['cv_results']['params'][p]))
         break

print
print('Reading test data file ...')
test = pd.read_csv('sign_mnist_test.csv')
print('Test shape: {}'.format(test.shape))

print
print('Scaling test dataset...')
X_test = test.loc[:, feature_columns]
y_test = test.loc[:, 'label']
scaler = MinMaxScaler(feature_range=(0,1))
X_test_scaled = scaler.fit_transform(X_test)

print
print('Predict test using best params...')
X_train = splitdata[9]['X_train']
y_train = splitdata[9]['y_train']

test_scores = {}
traintime = {}
predicttime = {}

ypred = {}
for classifier in classifiers:
  if classifier == 'AB':
    model = AdaBoostClassifier(n_estimators=40, learning_rate=0.1, random_state=seed)
  if classifier == 'DT':
    model = DecisionTreeClassifier(max_depth=20, max_features=1.0, random_state=seed)
  if classifier == 'KNN':
    model = KNeighborsClassifier(n_neighbors=1, weights='distance')
  if classifier == 'NN':
    model = MLPClassifier(max_iter=1000, hidden_layer_sizes=100, alpha=0.1, activation='tanh', random_state=seed)
  if classifier == 'SVM':
    model = SVC(kernel='linear', gamma=0.001, C=1, random_state=seed)
  start = timeit.default_timer()
  model.fit(X_train, y_train)
  traintime[classifier] = timeit.default_timer()-start
  start = timeit.default_timer()
  ypred[classifier] = model.predict(X_test)
  predicttime[classifier] = timeit.default_timer()-start
  score = accuracy_score(y_test, ypred[classifier])
  print('{} test score: {:.3f}'.format(classifier, score))
  test_scores[classifier] = score

  
setup_plot_2bar(validation_scores, test_scores, ['validation','test'], None, 'Accuracy Score', 'sl-valid-test.png')
setup_plot_2bar(traintime, predicttime, ['train','predict'], None, 'Run Time (s)', 'sl-train-predict-time.png')

print
print("==========")
print('End: {}'.format(strftime('%c', localtime())))
print("==========")
print('Total Time: {} secs'.format(int(timeit.default_timer()-pstart)))
