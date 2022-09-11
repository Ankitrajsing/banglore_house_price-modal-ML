# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 06:17:19 2022

@author: 91810
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1=pd.read_csv("Bengaluru_House_Data.csv")
# print(df1.head())
df2=df1.drop(['area_type','availability','balcony','society'],axis='columns')
# print(df2.head())
# print(df2.isnull().sum())
df3=df2.dropna()
# print(df3.isnull().sum())
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df3)
df3=df3.drop(['size'],axis='columns')
# print(df3)
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
# print(df3[~df3['total_sqft'].apply(is_float)].head())
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)    
# print(df4.head())
df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
# print(df5.head())
# print(len(df5.location.unique()))
df5.location = df5.location.apply(lambda x:x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=True)
print(location_stats)
location_stats_less_than_10 = location_stats[location_stats<=10]
df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# print(df5.head(10))
df6=df5[~(df5.total_sqft/df5.bhk<300)]
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m= np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out   
df7= remove_pps_outliers(df6)
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
                
                }
            for bhk,bhk_df in location_df.groupby('bhk'):
                stats=bhk_stats.get(bhk-1)
                if stats and stats['count']>5:
                    exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
        return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
df9= df8[df8.bath<df8.bhk+2] 
print(df9.head())  









































df10=df9.drop(['price_per_sqft'],axis='columns')
dummies=pd.get_dummies(df10.location)
df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df12=df11.drop('location',axis='columns')
x=df12.drop('price',axis='columns')
y=df12.price
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
cross_val_score(LinearRegression(),x,y,cv=cv)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos={
        'linear_regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
                }
            },
        'Lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']}
                },
        'decision_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
                }
            }
        }
    scores= []
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model':algo_name,
            'best_score':gs.best_score,
            'best_params':gs.best_params_
            })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    
    X=np.zeros(len(x.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    return lr_clf.predict([X])[0]  
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f) 
import json 
columns={
    'data_columns':[col.lower() for col in x.columns]}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
    




































