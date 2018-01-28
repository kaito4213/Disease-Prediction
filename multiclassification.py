'''
  gui_multilabel.py
  Copyright @ 2018 Jiaoyan<jchen11@wpi.edu>, Ziqi<zlin3@wpi.edu>, Han <hjiang@wpi.edu>
  License: MIT

'''

import scipy
from scipy import sparse
from sklearn.datasets import make_multilabel_classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

from skmultilearn.adapt import MLkNN
import datetime

def multilabel(df):
    X = df[['visiting_time_before', 'SP_ALZHDMTA',
       'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN',
       'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA',
       'AGE','BENE_SEX_IDENT_CD_1.0', 'BENE_SEX_IDENT_CD_2.0', 'BENE_RACE_CD_1.0',
       'BENE_RACE_CD_2.0', 'BENE_RACE_CD_3.0', 'BENE_RACE_CD_5.0',
       'BENE_ESRD_IND_0', 'BENE_ESRD_IND_Y', 'SP_STATE_CODE_1.0',
       'SP_STATE_CODE_2.0', 'SP_STATE_CODE_3.0', 'SP_STATE_CODE_4.0',
       'SP_STATE_CODE_5.0', 'SP_STATE_CODE_6.0', 'SP_STATE_CODE_7.0',
       'SP_STATE_CODE_8.0', 'SP_STATE_CODE_9.0', 'SP_STATE_CODE_10.0',
       'SP_STATE_CODE_11.0', 'SP_STATE_CODE_12.0', 'SP_STATE_CODE_13.0',
       'SP_STATE_CODE_14.0', 'SP_STATE_CODE_15.0', 'SP_STATE_CODE_16.0',
       'SP_STATE_CODE_17.0', 'SP_STATE_CODE_18.0', 'SP_STATE_CODE_19.0',
       'SP_STATE_CODE_20.0', 'SP_STATE_CODE_21.0', 'SP_STATE_CODE_22.0',
       'SP_STATE_CODE_23.0', 'SP_STATE_CODE_24.0', 'SP_STATE_CODE_25.0',
       'SP_STATE_CODE_26.0', 'SP_STATE_CODE_27.0', 'SP_STATE_CODE_28.0',
       'SP_STATE_CODE_29.0', 'SP_STATE_CODE_30.0', 'SP_STATE_CODE_31.0',
       'SP_STATE_CODE_32.0', 'SP_STATE_CODE_33.0', 'SP_STATE_CODE_34.0',
       'SP_STATE_CODE_35.0', 'SP_STATE_CODE_36.0', 'SP_STATE_CODE_37.0',
       'SP_STATE_CODE_38.0', 'SP_STATE_CODE_39.0', 'SP_STATE_CODE_41.0',
       'SP_STATE_CODE_42.0', 'SP_STATE_CODE_43.0', 'SP_STATE_CODE_44.0',
       'SP_STATE_CODE_45.0', 'SP_STATE_CODE_46.0', 'SP_STATE_CODE_47.0',
       'SP_STATE_CODE_49.0', 'SP_STATE_CODE_50.0', 'SP_STATE_CODE_51.0',
       'SP_STATE_CODE_52.0', 'SP_STATE_CODE_53.0', 'SP_STATE_CODE_54.0']]

    y = df[['ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
       'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7',
       'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']]

    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    classifier = MLkNN(k=20)
    classifier.fit(X_train, y_train)

    return classifier

def test_multilabel(classifier, X_test):
    predictions = classifier.predict(X_test)
    predictions = predictions.toarray()[0,:].tolist()
    disease = ['ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
       'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7',
       'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']
    pred_disease = []
    for i in range(len(predictions)):
        if predictions[i]==1:
            pred_disease.append(disease[i])
    output_result = "You have high risk of getting the following disease(s): {}".format(pred_disease)
    return output_result


