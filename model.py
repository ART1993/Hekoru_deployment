import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv("housing.csv",index_col=0)
df.drop(columns=["Alley","PoolQC","Fence",
                        "MiscFeature","FireplaceQu"],inplace=True)
df.dropna(inplace=True)
df.corr()[df.columns[-1]].where(abs(df.corr()[df.columns[-1]])>0.5).dropna(inplace=True)
for c in df.columns:
    if(np.dtype(df[c])==object and len(df[c].unique())<=1):
        df.drop(columns=[c],inplace=True)

def cleanframe(df,p_df):
    for c in df.columns[1:-1]:
        if(np.dtype(p_df[c])==object or c in ['MSSubClass','OverallQual',
        'OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']):
            p_df.loc[:,c] = p_df.loc[:,c].astype('category')
            arg_dummie = pd.get_dummies(p_df[c],prefix=c,prefix_sep='_')
            p_df.loc[:,arg_dummie.columns]=arg_dummie
            p_df.drop(columns=[c,arg_dummie.columns[-1]],inplace=True)

X = df.loc[:,df.columns[1:-1]]
Y = df.loc[:,df.columns[-1]]

cleanframe(df,X)

df_dummied=X
df_dummied["SalePrice"]=Y
df_dummied.drop(columns=['TotRmsAbvGrd','GarageCars'], inplace=True)
corr_df=df_dummied.corr()[df_dummied.columns[-1]].where(abs(df_dummied.corr()[df_dummied.columns[-1]])>0.6).dropna()

corr_X={}
for c in corr_df.index[:-1]:
    serie_Xc=X.corr()[c].where(abs(X.corr()[c])>0.8).dropna()
    serie_Xc=serie_Xc.where(serie_Xc<1.0).dropna()
    if len(serie_Xc>0):
        corr_X[c]=serie_Xc

X=df_dummied.loc[:,corr_df.index[:-1]]
Y=df_dummied.loc[:,'SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size = 0.3, random_state = 100)
 
regressor = LinearRegression()
regressor.fit(X_train.values, y_train.values)
y_pred = regressor.predict(X_test.values)

pickle.dump(regressor, open('model.pkl','wb'))




