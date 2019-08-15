import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.8545725108225108
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=LinearSVC(C=5.0, dual=True, loss="hinge", penalty="l2", tol=1e-05)),
        make_union(
            FunctionTransformer(copy),
            make_union(
                RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=1.0, n_estimators=100), step=0.9500000000000001),
                FunctionTransformer(copy)
            )
        )
    ),
    StandardScaler(),
    LinearSVC(C=0.001, dual=True, loss="hinge", penalty="l2", tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
