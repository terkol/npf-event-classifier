import numpy as np
import pandas as pd

s = pd.read_csv('triple_scores5.csv')
ens = s['ensemble'].to_numpy()
ens = [s.split('+') for s in ens]
for i in range(len(ens)):
    for j in range(len(ens[i])):
        if ens[i][j] == 'log':
            ens[i][j] = 'LR'
        elif ens[i][j] == 'lda':
            ens[i][j] = 'LDA'
        elif ens[i][j] == 'nb':
            ens[i][j] = 'NB'
        elif ens[i][j] == 'knn':
            ens[i][j] = 'K-NN'
        elif ens[i][j] == 'svc':
            ens[i][j] = 'SVM'
        elif ens[i][j] == 'rf':
            ens[i][j] = 'RF'
        elif ens[i][j] == 'hgb':
            ens[i][j] = 'HistGB'
        elif ens[i][j] == 'lgb':
            ens[i][j] = 'LightGBM'
        elif ens[i][j] == 'xgb':
            ens[i][j] = 'XGBoost'
        elif ens[i][j] == 'cgb':
            ens[i][j] = 'CatBoost'
ens = [sorted(s) for s in ens]
ens = ["+".join(s) for s in ens]
ens = [f'"{s}",' for s in ens]
s['ensemble'] = ens
s = s.set_index('ensemble').groupby(level=0).mean()
s = s.sort_values('score', ascending=False)
s.to_csv('triples_sorted.csv')
