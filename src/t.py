import numpy as np
import pandas as pd
u = pd.Series()
a = list(range(7))
for i in a:
    for j in a:
        for k in a: 
            u.loc[len(u)] = sorted([i,j,k])
print(u.unique())
