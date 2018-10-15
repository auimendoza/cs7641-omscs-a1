def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import timeit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
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
    learnercv = GridSearchCV(learner, params, scoring=make_scorer(f1_score))
    learnercv.fit(X_train, y_train)    
    results = {
        'best_score': learnercv.best_score_,
        'best_params': learnercv.best_params_,
        'best_estimator': learnercv.best_estimator_,
        'mean_fit_time': learnercv.cv_results_['mean_fit_time'][learnercv.best_index_],
        'mean_score_time': learnercv.cv_results_['mean_score_time'][learnercv.best_index_],
        'n_training': X_train.shape[0],
        'n_test': X_test.shape[0],
        'cv_results': learnercv.cv_results_ 
    }
    #print('{} done.'.format(label))
    #print('{} CV best f1 score: {}'.format(label, learnercv.best_score_))
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
    ax1.set_ylabel('F1 Score')
    
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
    fig.savefig(''.join(['lc-credit-card-',label,'-',str(paramidx),'.png']), bbox_inches="tight")
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
    ax1.set_ylabel('F1 Score')
    
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
    fig.savefig(''.join(['mc-credit-card-',label,'-',str(dataidx),'-',pisuffix,'.png']), bbox_inches="tight")
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
    
def setup_plot_2bar(data1, data2, legends, ylim, ylabel, filename):
  width=0.8
  vals = []

  xlabels = sorted(data1.keys())
  val1 = []
  for xl in xlabels:
      val1.append(data1[xl])
  val2 = []
  for xl in xlabels:
      val2.append(data2[xl])
  vals = [val1, val2]
  plot_valid_vs_test(xlabels, vals, legends, ylim, ylabel, filename)

print
print('Reading file...')
ccdata = pd.read_csv('creditcard.csv')

fraudtr = ccdata.loc[ccdata['Class'] == 1,:]
normaltr = ccdata.loc[ccdata['Class'] == 0,:]
fraudcnt = fraudtr.shape[0]
normalcnt = normaltr.shape[0]
print('fraudlent transactions count = {}'.format(fraudcnt))
print('normal transactions count = {}'.format(normalcnt))
print('percentage of fraudulent transactions {:.3f}%'.format(fraudcnt*100./(normalcnt+fraudcnt)))

seed=0

print
print('Upsampling data...')
upsample_total = int(fraudcnt/0.05)
normal_count = upsample_total-fraudcnt
normal_data = normaltr.sample(n=normal_count, random_state=seed)
upsample_data = pd.concat([fraudtr, normal_data])
print('upsample size = {}'.format(upsample_total))

seed = 0
feature_columns = ccdata.columns[:-1]
print(feature_columns)
print('feature columns count: {}'.format(len(feature_columns)))

print
print('Splitting data for each dataset size...')
splitdata = []
percsamples = list(np.arange(10,101,10)/100.)
for p in percsamples:
    n = int(upsample_total*p)
    Xy = upsample_data.sample(n=n, random_state=seed)
    print('dataset size {}: {}, %fraud: {:.3f}'.format(len(splitdata)+1, Xy.shape[0],Xy['Class'].value_counts(sort=True).map(lambda x: x*100./Xy.shape[0])[1]))
    X = Xy.loc[:, feature_columns]
    y = Xy.loc[:,['Class']]
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=seed)
    splitdata.append({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    })

print
print('Setting parameters...')
params = {}
params['DT'] = {'max_depth': [None, 5,6,7,8,9,10,11,12,13,14,15]}
params['KNN'] = {'n_neighbors': [3,5,7,10,12,15], 'p': [1], 'weights': ['distance']}
params['SVM'] = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100], 
                 'gamma': [1], 
                 'kernel': ['poly']}
params['NN'] = {'activation': ['logistic', 'relu'], 
                'solver': ['lbfgs'], 
                'learning_rate': ['constant'],
                'hidden_layer_sizes': range(10,101,10),
                'alpha': [0.1]
               }
params['AB'] = {'n_estimators': range(10,101,10),
                'learning_rate': [0.1, 1.0]
               }


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

n_datasets = len(percsamples)
for classifier in classifiers:
    #print('{} done.'.format(classifier))
    cvparams = results[classifier][0]['cv_results']['params']
    for i in range(len(results[classifier][0]['cv_results']['params'])):
        if (classifier == 'DT' and cvparams[i]['max_depth'] is not None):
            continue
        if (classifier == 'AB' and 
            cvparams[i]['n_estimators'] != 100):
            continue
        if (classifier == 'KNN' and 
            cvparams[i]['n_neighbors'] != 3):
            continue
        if (classifier == 'NN' and
            cvparams[i]['hidden_layer_sizes'] != 100):
            continue
        if (classifier == 'SVM' and
            cvparams[i]['C'] != 0.5
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
        #print('--- use params: [{}]{} ---'.format(useindex[classifier], results[classifier][0]['cv_results']['params'][useindex[classifier]]))
        for n in range(n_datasets):
            nsizeout = results[classifier][n]
            cvresults = nsizeout['cv_results']
            lcurve_x.append(nsizeout['n_training']+nsizeout['n_test'])
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
    #print('{} done.'.format(classifier))
    hyperparam = hyperparams[classifier]
    #for dataidx in range(n_datasets):
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
                    (rparams['activation'] == 'tanh' or
                     rparams['learning_rate'] == 'invscaling' or
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
                #print('paramindx', pi)
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
            paramtext = "n = {}, {}".format(ciresults['n_training']+ciresults['n_test'], u)
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
print('Predict test using best params...')
X_train = splitdata[9]['X_train']
y_train = splitdata[9]['y_train']
X_test = splitdata[9]['X_test']
y_test = splitdata[9]['y_test']

test_scores = {}
traintime = {}
predicttime = {}

ypred = {}
for classifier in classifiers:
  if classifier == 'AB':
    model = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=seed)
  if classifier == 'DT':
    model = DecisionTreeClassifier(max_depth=5, random_state=seed)
  if classifier == 'KNN':
    model = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1)
  if classifier == 'NN':
    model = MLPClassifier(max_iter=1000, hidden_layer_sizes=100, alpha=0.1, activation='relu', solver='lbfgs', learning_rate='constant', random_state=seed)
  if classifier == 'SVM':
    model = SVC(kernel='poly', gamma=1, C=0.5, random_state=seed)

  start = timeit.default_timer()
  model.fit(X_train, y_train)
  traintime[classifier] = timeit.default_timer()-start

  start = timeit.default_timer()
  ypred[classifier] = model.predict(X_test)
  predicttime[classifier] = timeit.default_timer()-start

  score = f1_score(y_test, ypred[classifier])
  print('{} test score: {:.3f}'.format(classifier, score))
  test_scores[classifier] = score

  
setup_plot_2bar(validation_scores, test_scores, ['validation','test'], [0.8, 1.], 'F1 Score', 'cc-valid-test.png')
setup_plot_2bar(traintime, predicttime, ['train','predict'], [0., 1.1], 'Run Time (s)', 'cc-train-predict-time.png')

from sklearn.metrics import confusion_matrix

print
for classifier in classifiers:
    print("=== {} confusion matrix ===".format(classifier))
    tn, fp, fn, tp =  confusion_matrix(y_test, ypred[classifier]).ravel()
    print("true  positive: {}".format(tp))
    print("false positive: {}".format(fp))
    print("true  negative: {}".format(tn))
    print("false negative: {}".format(fn))

print
print("==========")
print('End: {}'.format(strftime('%c', localtime())))
print("==========")
print('Total Time: {} secs'.format(int(timeit.default_timer()-pstart)))
