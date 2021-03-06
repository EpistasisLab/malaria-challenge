�csklearn.pipeline
Pipeline
q )�q}q(X   stepsq]q(X   featureunionqcsklearn.pipeline
FeatureUnion
q)�q}q(X   transformer_listq	]q
(X
   pipeline-1qh )�q}q(h]q(X   standardscalerqcsklearn.preprocessing.data
StandardScaler
q)�q}q(X	   with_meanq�X   with_stdq�X   copyq�X   _sklearn_versionqX   0.21.2qub�qX   selectpercentileqcsklearn.feature_selection.univariate_selection
SelectPercentile
q)�q}q(X
   score_funcqcsklearn.feature_selection.univariate_selection
f_classif
qX
   percentileqKQhhub�q eX   memoryq!NX   verboseq"�hhub�q#X
   pipeline-2q$h )�q%}q&(h]q'(X   standardscalerq(h)�q)}q*(h�h�h�hhub�q+X   rfeq,csklearn.feature_selection.rfe
RFE
q-)�q.}q/(X	   estimatorq0csklearn.ensemble.forest
ExtraTreesClassifier
q1)�q2}q3(X   base_estimatorq4csklearn.tree.tree
ExtraTreeClassifier
q5)�q6}q7(X	   criterionq8X   giniq9X   splitterq:X   randomq;X	   max_depthq<NX   min_samples_splitq=KX   min_samples_leafq>KX   min_weight_fraction_leafq?G        X   max_featuresq@X   autoqAX   random_stateqBNX   max_leaf_nodesqCNX   min_impurity_decreaseqDG        X   min_impurity_splitqENX   class_weightqFNX   presortqG�hhubX   n_estimatorsqHKdX   estimator_paramsqI(h8h<h=h>h?h@hChDhEhBtqJX	   bootstrapqK�X	   oob_scoreqL�X   n_jobsqMNhBMZh"K X
   warm_startqN�hFNh8h9h<Nh=Kh>Kh?G        h@G?�fffffghCNhDG        hENhhubX   n_features_to_selectqONX   stepqPG?陙����h"K hhub�qQeh!Nh"�hhub�qRehMNX   transformer_weightsqSNh"�hhub�qTX	   linearsvcqUcsklearn.svm.classes
LinearSVC
qV)�qW}qX(X   dualqY�X   tolqZG>�����h�X   Cq[G?PbM���X   multi_classq\X   ovrq]X   fit_interceptq^�X   intercept_scalingq_KhFNh"K hBMZX   max_iterq`M�X   penaltyqaX   l2qbX   lossqcX   hingeqdhhub�qeeh!Nh"�hhub.