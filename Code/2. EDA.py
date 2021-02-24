# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:21:21 2019

@author: Mahantesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
Raw_Dataset = pd.read_csv('C:\\Users\\Mahantesh\Desktop\\Summer Sem 2019\\AML\\Project\\Raw Dataset\\Dataset.csv')
Labels =  Raw_Dataset.drop(columns = 'TEXT')


#Number of documents
len(Raw_Dataset)

#Number of labels
len(Labels.columns)

#--------------------------------------------------------------------------------------------------
# Average Labels per document

sum_Lables =0 
for i in range(3954):
    sum_Lables = sum_Lables + Labels.iloc[:,i].sum()

Avg_Labels = float(sum_Lables/len(Raw_Dataset))
#5.3

#--------------------------------------------------------------------------------------------------
df_Labels_freq =[]
col_names =[]


for i in range(3954):
    df_Labels_freq.append(Labels.iloc[:,i].sum())
    col_names.append(Labels.columns[i])
    

df_Labels_freq =pd.DataFrame(df_Labels_freq)
df_Labels_freq["Label"] = pd.Series(col_names)

df_Labels_freq.rename(columns = {0 : 'Count'}, inplace =True)

#Sort to see the labels with maximum occurence or frequency

Top_50 = df_Labels_freq.sort_values(by=['Count'],ascending = False).head(50)

Top_50['Count'].sum()/sum_Lables

#Top 50 labels occupy 23% of the total labels

#Top 100
Top_100 = df_Labels_freq.sort_values(by=['Count'],ascending = False).head(100)

Top_100['Count'].sum()/sum_Lables

Top_100.to_csv('C:\\Users\\Mahantesh\Desktop\\Summer Sem 2019\\AML\\Project\\Top_100.csv')

#---------------------------------------------------------------------------------------------------
#Counting the documents and labels correspondance, eg how many documents have 5 labels
doc_labels = pd.DataFrame(Labels.sum(axis=1))
doc_labels.rename(columns = {0 : 'Number of Labels'}, inplace =True)
doc_labels["Number of Labels"] = doc_labels["Number of Labels"].astype(str)
doc_labels["Count"] = 1

doc_labels_corr = pd.DataFrame( doc_labels.groupby(["Number of Labels"])["Count"].sum()).reset_index()

doc_labels_corr.to_csv('C:\\Users\\Mahantesh\Desktop\\Summer Sem 2019\\AML\\Project\\doc_labels_corr.csv', index = False)



#------------------------------------------------------------------------------------------------------
#look at the correlation between documents

def cosine_similarity(A,B):
    sum_vectors =0
    sum_A =0
    sum_B =0
    for i in range(len(A)):
        sum_vectors = sum_vectors +  A[i]*B[i]
        sum_A = sum_A + A[i]*A[i]
        sum_B = sum_B + B[i]*B[i]
    
    Sim = sum_vectors / (math.sqrt(sum_A)*math.sqrt(sum_B))
    return Sim

print(cosine_similarity(Labels.iloc[:,0], Labels.iloc[:,1]))


Similarity = []
for i in range(3954):
    print(i)
    Similarity.append(cosine_similarity(Labels.iloc[:,0],Labels.iloc[:,i]))
    print(cosine_similarity(Labels.iloc[:,0],Labels.iloc[:,i]))    




from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(np.array([1,1]), np.array([1,1])))
#--------------------------------------------------------------------------------------------------------

x = np.array([2,3,1,0])
y = np.array([2,3,0,0])

x = x.reshape(1,-1)
y = y.reshape(1,-1)

Co_Sim =[]
for w in range(len(Raw_Dataset.columns)-1):
    Co_Sim.append([])
    for k in range(len(Raw_Dataset.columns)-1):
        Co_Sim[w].append(cosine_similarity(np.array(Raw_Dataset.iloc[:,1]).reshape(1,-1),np.array(Raw_Dataset.iloc[:,k+1]).reshape(1,-1)) )
    #print(i)
     #dot = sum([i*j for (i, j) in zip(Raw_Dataset.iloc[:,1], Raw_Dataset.iloc[:,w+1])])
     if(dot>0):
         print(w)
     V1_sum = Raw_Dataset.iloc[:,1].sum()
     V2_sum = Raw_Dataset.iloc[:,i+1].sum()

     Co_Sim.append(2*dot/(V1_sum + V2_sum))
    #print(cosine_similarity(np.array(Raw_Dataset.iloc[:,1]).reshape(1,-1),np.array(Raw_Dataset.iloc[:,i+1]).reshape(1,-1)) )

pd.DataFrame(Co_Sim).sum()

from sklearn.metrics import confusion_matrix


Co_Similar_index =[]

for i in range(3954):
    Co_Similar_index.append([])

for w in range(len(Data_corr.columns)):
    for k in range(len(Data_corr.columns)):
        print(w)
        print(k)
        tn, fp, fn, tp = confusion_matrix(Data_corr.iloc[:,w],Data_corr.iloc[:,k]).ravel()
        
        if(tp>0):
            Co_Similar_index[w].append(k)
            #Co_Similar_Cols.append(Raw_Dataset.columns[k+1])


for i in range(len(Co_Sim)):
    if (Co_Sim[i][0][0] >0):
        print(i)
    
x =np.array(Raw_Dataset.iloc[:,1]).reshape(1,-1)
 (cosine_similarity(np.array(Raw_Dataset.iloc[:,1]).reshape(1,-1),np.array(Raw_Dataset.iloc[:,5]).reshape(1,-1)) )


#Now lets use some other metrics
 
#---------------------------------------------------------------------------------------------------------
 #Finally we look at only top 100 labels
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity


Data_corr = Raw_Dataset.iloc[:,Raw_Dataset.columns != 'TEXT'] 


labels_100 = pd.read_csv("C:\\Users\\Mahantesh\\Desktop\\Summer Sem 2019\\AML\\Project\\EDA\\Results\\Top_100_label_freq.csv")
 
labels_100.iloc[:,0]

Raw_Dataset.columns.get_loc(labels_100.iloc[0,0])

import math
Co_Similar_index =[]
Similarity = []
Similar_Cols_names = []

for i in range(len(labels_100)):
    Co_Similar_index.append([])
    Similarity.append([])
    Similar_Cols_names.append([])

for w in range(len(labels_100)):
      
    for k in range(len(Data_corr.columns)):
        print(w)
        print(k)
        tn, fp, fn, tp = confusion_matrix(Data_corr.iloc[:,Data_corr.columns.get_loc(labels_100.iloc[w,0])],Data_corr.iloc[:,k]).ravel()
        
        if(tp>0):
            Co_Similar_index[w].append(k)
            ad = tp*tn
            bc = fp*fn
            Similarity[w].append((ad-bc)*100/math.sqrt((tp+fp)*(tn +fn )*(tp+fn )*(tn+fp )))
            Similar_Cols_names[w].append(Data_corr.columns[k])



#Counting the number of labels that are correlated  

Sim_Column_names = []
Similarity_score = []
Sim_count =[]

for i in range(len(labels_100)):
    Similarity_score.append([])
    Sim_Column_names.append([])


for i in range(len(labels_100)):
    for k in range(len(Similarity[i])):
        if(Similarity[i][k]>0.8):
            Similarity_score[i].append(Similarity[i][k])
            Sim_Column_names[i].append(Similar_Cols_names[i][k])
    Sim_count.append(len(Similarity_score[i]))       


pd.DataFrame(Sim_count).to_csv('C:\\Users\\Mahantesh\Desktop\\Summer Sem 2019\\AML\\Project\\Top_100_labels_corr_count.csv', index = False)       

    
    if(Similarity[2][i] > 50):
        print(i)


#--------------------------------------------------------------------------------------------------
