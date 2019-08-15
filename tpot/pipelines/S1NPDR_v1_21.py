import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.8172868435911914
exported_pipeline = make_pipeline(
    make_union(
        StandardScaler(),
        RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.2, n_estimators=100), step=0.1)
    ),
    VarianceThreshold(threshold=0.25),
    StandardScaler(),
    StandardScaler(),
    MinMaxScaler(),
    StandardScaler(),
    LogisticRegression(C=0.01, dual=False, penalty="l2")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
