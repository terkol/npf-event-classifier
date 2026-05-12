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

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier

path = os.path.dirname(__file__)
scores = pd.read_csv('scores.csv')
# scores = pd.read_csv('scores3.csv')
# scores = pd.read_csv('scores4.csv')
scores = scores.set_index('test').sort_index()[::-1]

top_scores = scores.head(10)
m1 = top_scores['m1'].value_counts().sort_index()
m2 = top_scores['m2'].value_counts().sort_index()
# m3 = top_scores['m3'].value_counts().sort_index()
# m4 = top_scores['m4'].value_counts().sort_index()
m = m1.add(m2,fill_value=0)
# m = m.add(m3,fill_value=0)
# m = m.add(m4,fill_value=0)
print(m/sum(m))