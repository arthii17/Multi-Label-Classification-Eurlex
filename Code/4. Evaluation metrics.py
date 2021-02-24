# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:41:16 2019

@author: Mahantesh
"""

import pandas as pd
import numpy as np

br = pd.read_csv('C:\\Users\\Mahantesh\\Desktop\\Summer Sem 2019\\AML\\Project\\Predictions\\BinaryRelevance_tfidf_predictions.csv')
MlkNN = pd.read_csv('C:\\Users\\Mahantesh\\Desktop\\Summer Sem 2019\\AML\\Project\\Predictions\\MLknn_tfidf_predictions.csv')
valid_y = pd.read_csv("C:\\Users\Mahantesh\\Desktop\\Summer Sem 2019\\AML\\Project\\Dataset for Testing\\Valid_Y.csv")

#-----------------------------------------------------------------------------------------------------------
#Assign the model of your choice

predictions_thresh = np.add(np.array(br),np.array(MlkNN))
predictions_br_MlkNN = np.where(predictions_thresh > 0.4 , 1, 0)


predictions_br_MlkNN = pd.DataFrame(predictions_br_MlkNN)


model = MlkNN

#---------------------------------------------------------------------------------------------------------------------
#True Positives
df_tp=[]

for j in range(len(valid_y)):
    df_tp.append([])

for j in range(len(valid_y)):
    for i in range(3954):
        df_tp[j].append(0)

df_tp = pd.DataFrame(df_tp)
   
#df.rename(columns={0: 'tp'}, inplace=True)
    
for k in range(3954):
    df_tp.iloc[:,k] = 0
    df_tp.iloc[:,k] = np.where((model.iloc[:,k] == valid_y.iloc[:,k]) & (model.iloc[:,k]==1),1,0)
                    
tp = np.matrix(df_tp).sum()

#------------------------------------------------------------------------------------------------------------
#True Negatives
df_tn=[]

for j in range(len(valid_y)):
    df_tn.append([])

for j in range(len(valid_y)):
    for i in range(3954):
        df_tn[j].append(0)

df_tn = pd.DataFrame(df_tn)
    
for k in range(3954):
    df_tn.iloc[:,k] = 0
    df_tn.iloc[:,k] = np.where((model.iloc[:,k] == valid_y.iloc[:,k]) & (model.iloc[:,k]==0),1,0)
                    
tn = np.matrix(df_tn).sum()

#---------------------------------------------------------------------------------------------------------------
#False Positives
df_fp=[]

for j in range(len(valid_y)):
    df_fp.append([])

for j in range(len(valid_y)):
    for i in range(3954):
        df_fp[j].append(0)

df_fp = pd.DataFrame(df_fp)
    
for k in range(3954):
    df_fp.iloc[:,k] = 0
    df_fp.iloc[:,k] = np.where((model.iloc[:,k] != valid_y.iloc[:,k]) & (model.iloc[:,k]==1) & (valid_y.iloc[:,k]==0) ,1,0)
                    
fp = np.matrix(df_fp).sum()

#-----------------------------------------------------------------------------------------------------------------

#False Negatives
df_fn=[]

for j in range(len(valid_y)):
    df_fn.append([])

for j in range(len(valid_y)):
    for i in range(3954):
        df_fn[j].append(0)

df_fn = pd.DataFrame(df_fn)
    
for k in range(3954):
    df_fn.iloc[:,k] = 0
    df_fn.iloc[:,k] = np.where((model.iloc[:,k] != valid_y.iloc[:,k]) & (model.iloc[:,k]==0) & (valid_y.iloc[:,k]==1) ,1,0)
                    
fn = np.matrix(df_fn).sum()

#-------------------------------------------------------------------------------------------------------
#Micro Precision
Micro_Precision = tp*100/(tp+fp)

#Micro Recall
Micro_Recall = tp*100/(tp+fn)

#Micro F1 Score`
Micro_F1_Score = 2*Micro_Precision*Micro_Recall/(Micro_Precision+Micro_Recall)

#Accuracy
Accuracy = (tp+tn)*100/(tp+tn+fp+fn)

#Hamming Loss
Hamming_Loss = (fp+fn)*100/(tp+tn+fp+fn)

#-------------------------------------------------------------------------------------------------------
#Macro Average Precision

#Macro Precision
df_Macro_precision = []

for i in range(3954):
    df_Macro_precision.append(float(df_tp[i].sum()*100/(df_tp[i].sum() + df_fp[i].sum())))

Macro_precision = np.nanmean(np.matrix(df_Macro_precision))

#Macro Recall
df_Macro_recall = []

for i in range(3954):
        df_Macro_recall.append(float(df_tp[i].sum()*100/(df_tp[i].sum() + df_fn[i].sum())))

Macro_recall = np.nanmean(np.matrix(df_Macro_recall))


#Macro Average
df_Macro_average = []

for i in range(3954):
        df_Macro_average.append(float(   (df_tp[i].sum() + df_tn[i].sum() )*100/(df_tp[i].sum() + df_fn[i].sum()   +df_fp[i].sum()   +df_tn[i].sum()   )))

Macro_average = np.nanmean(np.matrix(df_Macro_average))

