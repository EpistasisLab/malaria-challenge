#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:36:53 2019

@author: aorlenko
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer
from copy import copy
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler, Binarizer, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif, RFE, SelectFwe
from tpot.builtins import StackingEstimator, ZeroCount, OneHotEncoder
from sklearn.kernel_approximation import RBFSampler
from tpot import TPOTClassifier
from sklearn.calibration import CalibratedClassifierCV
import csv
from sklearn.metrics import recall_score, make_scorer, SCORERS
tpot_obj = TPOTClassifier()

def balanced_accuracy(y_true, y_pred):
    all_classes = list(set(np.append(y_true, y_pred)))
    all_class_accuracies = []
    for this_class in all_classes:
        this_class_sensitivity = 0.
        this_class_specificity = 0.
        if sum(y_true == this_class) != 0:
            this_class_sensitivity = \
                float(sum((y_pred == this_class) & (y_true == this_class))) /\
                float(sum((y_true == this_class)))

            this_class_specificity = \
                float(sum((y_pred != this_class) & (y_true != this_class))) /\
                float(sum((y_true != this_class)))

        this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
        all_class_accuracies.append(this_class_accuracy)

    return np.mean(all_class_accuracies)

##### testing set filtering ######


####### loading training set ################
training = pd.read_csv('/PATH/Subset_npdred.csv', delimiter=',')
Y_CV = training['ClearanceRate']
training.drop('ClearanceRate', axis=1, inplace = True)
columns_train = training.columns
print('training test shape', training.shape)
X_CV = training


####### loading testing set ################
testing = pd.read_csv('/PATH/SubCh2_TestData.csv', delimiter=',')
test_SN = testing.Sample_Names.reset_index(drop=True)
test_all = testing[columns_train].reset_index(drop=True)

#### testing dataset colunmns filtered
testing_all = testing[columns_train].reset_index(drop=True)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state = 42)

### list with selected piepline numbers ######
with open('/PATH/pipelines/pipe_list.txt') as f:
    pipelines = f.read().splitlines()


### this will give us a piepline BA score, class value and decision function value #########           
def give_me_class_value(N, testset):
    ##### N is a pipeline number from the list ########
    clf = N
    tpot_obj._set_param_recursive(clf.steps, 'random_state', 4442)
    ######### Cross-validated balanced accuracy ##########
    scores2 =  cross_val_score(clf, X_CV, Y_CV, cv=cv,scoring=make_scorer(balanced_accuracy))    
    meanBA = np.mean(scores2)
    clf.fit(X_CV, Y_CV)
    class_value = clf.predict(testset)

    return meanBA, class_value

list_BA=[]
class_value_df = pd.DataFrame()

##### going through preselected pipelines with top 20 BA score ##########
for i in pipelines:
    print(i)
    ### loading a model from the preselected list######
    loaded_model = pickle.load(open('/PATH/pipelines/S1NPDR_v1_pickle_'+i+'.py', 'rb'))
    ####### insert the test set #########
    a, b, c = give_me_class_value(loaded_model, testing_all)
    list_BA.append(a)
    class_value_df[i] = b

print('overall BA score', np.mean(list_BA))  
  
##### saving BA scores ###########    
with open('/PATH/list_BA.txt','w') as f:
    for item in list_BA:
        f.write("%s\n" % item)



class_value_df['Predicted_Categorical_Clearance'] = class_value_df.mean(axis=1)
class_value_df['Probability'] = class_value_df.mean(axis=1)

##### turning 'mean' into binary by applying 0.5 threshold ########
a = np.array(class_value_df['Predicted_Categorical_Clearance'].values.tolist())
class_value_df['Predicted_Categorical_Clearance'] = np.where(a<0.5,0.0,a).tolist()
a = np.array(class_value_df['Predicted_Categorical_Clearance'].values.tolist())
class_value_df['Predicted_Categorical_Clearance'] = np.where(a>0.5,1.0,a).tolist()

class_value_df['Sample_Names'] = test_SN
class_value_df.to_csv('/PATH_OUT/testing_set_class_values_all.csv', index=False)


