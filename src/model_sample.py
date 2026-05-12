import numpy as np
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd


path = os.path.dirname(__file__)
train = pd.read_csv(path+'\\train.csv')
train = train.drop(columns=['date', 'partlybad'])
test = pd.read_csv(path+'\\test.csv')
test = test.drop(columns=['date', 'partlybad'])
X_test = test.set_index('id')
X = train.drop(columns='class4').set_index('id')
y = train['class4']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
        )

EVENT_CLASSES = np.array(["II","Ia","Ib"])
NONEVENT_CLASS = "nonevent"

# fit on full training
pipe.fit(X, y)

# predict on test
proba = pipe.predict_proba(X_test)
classes = pipe.named_steps["logisticregression"].classes_

# class2 must be P(event)
p_event = proba[:, np.isin(classes, EVENT_CLASSES)].sum(axis=1)

# class4 is the predicted 4-class label (argmax)
class4_pred = classes[np.argmax(proba, axis=1)]

# build submission exactly like Kaggle expects
sample = pd.read_csv("submission.csv")
sample["p"] = p_event
sample["class4"] = class4_pred
sample.to_csv(path + "\\ssubmission.csv", index=False)
