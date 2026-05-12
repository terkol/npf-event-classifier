import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, psutil

from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
from sklearn.feature_selection import SequentialFeatureSelector

path = os.path.dirname(__file__)
train = pd.read_csv(path+'\\train.csv')
train = train.drop(columns=['date', 'partlybad'])
train['class4'] = train['class4'].map({'II': 3, 'Ia': 2, 'Ib': 1, 'nonevent': 0})
test = pd.read_csv(path+'\\test.csv')
test = test.drop(columns=['date', 'partlybad'])
X_test = test.set_index('id')
X = train.drop(columns='class4').set_index('id')
y = train['class4']

EVENT_CLASSES = np.array([3, 2, 1])
NONEVENT_CLASS = 0

def filter_gaussian(X):
    gaussian = X.iloc[:,[0,2,4,6,10,12,14,16,18,20,22,47,50,52,54,56,58,74,84,86,88,90,92]]
    return gaussian

def drop_outliers(X, y, lim=0.5):
    NO_mask = X['NO168.mean'] < lim
    X = X.loc[NO_mask]
    y = y.loc[NO_mask]
    return X, y

def top_corr_features(X):
    top_corr = ['Glob.mean', 'Glob.std', 'NET.mean', 'NET.std', 'O3168.mean',
       'O342.mean', 'O3504.mean', 'O3672.mean', 'O384.mean', 'PAR.mean',
       'PAR.std', 'PTG.std', 'RGlob.mean', 'RGlob.std', 'RHIRGA168.mean',
       'RHIRGA168.std', 'RHIRGA336.mean', 'RHIRGA336.std', 'RHIRGA42.mean',
       'RHIRGA42.std', 'RHIRGA504.mean', 'RHIRGA504.std', 'RHIRGA672.mean',
       'RHIRGA672.std', 'RHIRGA84.mean', 'RHIRGA84.std', 'T168.std', 'T42.std',
       'T504.std', 'T672.std', 'T84.std', 'UV_A.mean', 'UV_A.std', 'UV_B.mean',
       'UV_B.std']
    # top_corr = ['Glob.mean','PAR.mean','RHIRGA168.mean','RHIRGA336.mean','RHIRGA42.mean','RHIRGA42.std','RHIRGA504.mean','RHIRGA672.mean','RHIRGA84.mean']
    X = X[top_corr]
    return X

def polynomial(X, deg=2, i_o=True):
    X = PolynomialFeatures(deg, interaction_only=i_o).fit_transform(X)
    return X

def perplexity_scorer(estimator, X, y):
    y_proba = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    event_idx = np.isin(classes, EVENT_CLASSES)
    p_event = y_proba[:, event_idx].sum(axis=1)
    eps = 1e-4
    p_event = np.clip(p_event, eps, 1-eps)
    y_bin = np.isin(y, EVENT_CLASSES).astype(int)
    bll = log_loss(y_bin, p_event)
    return np.exp(bll)

def binary_accuracy_scorer(estimator, X, y):
    y_pred = np.asarray(estimator.predict(X))
    y_true = np.asarray(y)
    y_pred_bin = np.isin(y_pred, EVENT_CLASSES).astype(int)
    y_true_bin = np.isin(y_true, EVENT_CLASSES).astype(int)
    return accuracy_score(y_true_bin, y_pred_bin)

scoring = {
    "perplexity": perplexity_scorer,
    "binary_accuracy": binary_accuracy_scorer,
    "multi_lable_accuracy": 'accuracy'
    }

def test_model(X, y, pipeline, n=0):
    if n != 0: print(n)
    # X, y = drop_outliers(X, y)
    # X = top_corr_features(X)
    # X = polynomial(X, i_o=True)
    # X = filter_gaussian(X)

    # X = X.iloc[:,:10]

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    # pipeline = CalibratedClassifierCV(pipeline, cv=cv, method='isotonic')
    cval = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, return_train_score=True)

    train_ma = np.mean(cval['train_multi_lable_accuracy'])
    train_ba = np.mean(cval['train_binary_accuracy'])
    train_per = np.mean(cval['train_perplexity'])
    per = np.mean(cval['test_perplexity'])
    ba = np.mean(cval['test_binary_accuracy'])
    ma = np.mean(cval['test_multi_lable_accuracy'])

    if n == -1:
        # print('train_perplexity          ', train_per)
        # print('train_binary_accuracy     ', train_ba)
        # print('train_multi_lable_accuracy', train_ma)
        print('test_perplexity          ', per)
        print('test_binary_accuracy     ', ba)
        print('test_multi_lable_accuracy', ma)

    return ba, ma, per
    return  (train_ba+train_ma+max(0,min(1,2-train_per)))/3, (ba+ma+max(0,min(1,2-per)))/3

def return_pipeline(n):
    pipe = make_pipeline( 
        # SelectKBest(mutual_info_classif, k=30),
        # StandardScaler(), 
        # PCA(n_components=n),
        # SequentialFeatureSelector(GaussianNB(), n_jobs=-1, tol=0.1),
        GaussianNB()
        )
    return pipe


log = make_pipeline(StandardScaler(), LogisticRegression(C=0.2))
lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(tol=0.28))
nb = make_pipeline(PCA(n_components=9), GaussianNB())
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25))
svc = make_pipeline(StandardScaler(), SVC(probability=True, C=3))
rf = make_pipeline(RandomForestClassifier(max_depth=6))
hgb = make_pipeline(HistGradientBoostingClassifier(max_depth=1))
lgb = make_pipeline(LGBMClassifier(verbosity=-1,n_estimators=30, colsample_bytree=0.1, reg_lambda=1))
cgb = make_pipeline(CatBoostClassifier(n_estimators=30, max_depth=4, verbose=False))
xgb = make_pipeline(XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=1))

clf = {'log':log, 'lda': lda, 'nb': nb, 'knn': knn, 'svc':svc, 'rf':rf, 'hgb': hgb, 'lgb':lgb, 'cgb': cgb, 'xgb':xgb}

# scores = []
# for n in np.arange(100):
#     pipe = VotingClassifier(estimators=[list(clf.items())[3], list(clf.items())[1]], voting="soft")
#     scores.append([n, *test_model(X, y, pipe, n)])
# agg_scores = [(i[1]+i[2]+max(0,min(1,2-i[3])))/3 for i in scores]
# scores = np.array(scores)
# # print(f'Average training score: {scores[:,1].mean()}')
# print(f'Average test score:     {np.mean(agg_scores)}')
# print(f'Deviation: {np.std(agg_scores)}')
# print(f'Average ba:     {scores[:,1].mean()}')
# print(f'Average ma:     {scores[:,2].mean()}')
# print(f'Average per:     {scores[:,3].mean()}')

# comb2 = [(1,6),(1,8),(1,9),(1,4),(1,7),(1,3),(1,5),(0,4),(1,0),(0,6),(0,5),(0,8),(0,9),(0,7),(6,8)]
# for c in comb2:
#     scores = []
#     for n in np.arange(50):
#         pipe = VotingClassifier(estimators=[list(clf.items())[c[0]], list(clf.items())[c[1]]],voting="soft")
#         scores.append([n, *test_model(X, y, pipe, n)])
#     scores = np.array(scores)
#     # print(scores)
#     print(c)
#     print(f'Average training score: {scores[:,1].mean()}')
#     print(f'Average test score:     {scores[:,2].mean()}')

# comb3 = [(1,7,4),(1,8,3),(1,4,6),(1,0,6),(1,0,9),(1,0,8),(1,9,4),(1,8,4),(1,0,4),(1,6,8),(1,7,8),(1,7,3),(1,6,4),(1,6,5),(1,3,6)]
# for c in comb3:
#     scores = []
#     for n in np.arange(50):
#         pipe = VotingClassifier(estimators=[list(clf.items())[c[0]], list(clf.items())[c[1]], list(clf.items())[c[2]]],voting="soft")
#         scores.append([n, *test_model(X, y, pipe, n)])
#     scores = np.array(scores)
#     # print(scores)
#     print(c)
#     print(f'Average training score: {scores[:,1].mean()}')
#     print(f'Average test score:     {scores[:,2].mean()}')
i=0
s = pd.DataFrame(columns=['ensemble', 'score'])
for (k,v) in clf.items():
    for (l,w) in clf.items():
        for (m,x) in clf.items():
            if k!=l and k!=m and l!=m:
                print(i)
                i += 1
                scores = []
                for _ in range(2):
                    pipe = VotingClassifier(estimators=[(k, v), (l, w), (m, x)], voting="soft")
                    scores.append(test_model(X, y, pipe)[1])
                s.loc[len(s)] = [f'{k}+{l}+{m}', np.mean(scores)]
s.to_csv('triple_scores2.csv')

