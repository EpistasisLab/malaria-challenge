from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from tpot.config.classifier_nn import classifier_config_nn
from sklearn.pipeline import make_pipeline
# from tpot.config import classifier_config_dict_light
from tpot.config import classifier_config_dict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics.scorer import make_scorer

# personal_config = classifier_config_dict_light
personal_config = classifier_config_dict


accuracy_ls = []
n_gen = 100
n_pop = 100

tpot_data = pd.read_csv('../data/SubCh2/SubCh2_TrainingData_for_tpot.csv')
Xdata = tpot_data.loc[:, tpot_data.columns != 'ClearanceRate']
Xdata = Xdata.drop(Xdata.columns[0:3], axis=1)
Ydata = tpot_data['ClearanceRate']

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata.values, random_state = 1618,
                                                    train_size=0.75, test_size=0.25)

del Xdata
del Ydata
del tpot_data

for seed in range(100):
    tpot = TPOTClassifier(generations=n_gen, config_dict=personal_config,
                          population_size=n_pop, verbosity=2, random_state=seed,
                          early_stop=20, scoring='average_precision',
                          template='Selector-Transformer-Classifier')
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))

    tpot.export('pipelines/npdr_1_' + str(seed) + '.py')
    accuracy_ls.append([tpot._optimized_pipeline_score, tpot.score(X_test, y_test)])
    accuracy_mat = pd.DataFrame(accuracy_ls, columns = ['Training CV Accuracy', 'Testing Accuracy'])
    accuracy_mat.to_csv("accuracies/" + str(n_gen) + '_' + str(n_pop) + '_' + str(seed) + ".tsv", sep = "\t")
