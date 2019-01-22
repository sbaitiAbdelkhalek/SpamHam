import app
import pandas as pd
import numpy as np

labelsName= {"ham":0,"spam":1}

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df = df.replace(['ham','spam'],[labelsName["ham"], labelsName["spam"]])
df = df.rename({"v1":"label","v2":"message"},axis=1)

df['Count']=0
for i in np.arange(0,len(df.message)):
    df.loc[i,'Count'] = len(df.loc[i,'message'])



corpus = df.message
lables = df.label
numericCorpus = app.preparingDataForModel(corpus)

model, rapport = app.trainingmodel(numericCorpus,lables,labelsName)

print(rapport)