'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月22日 14:54
@Description: 
@URL: https://github.com/oskird/Kaggle-Home-Credit-Default-Risk-Solution/blob/master/Feature%20Selection.ipynb
@version: V1.0
'''
#Prepare
import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings("ignore")

application_train = pd.read_csv( './data/application_train.csv' )

#Stratified Sampling (ratio = 0.1)
application_sample1 = application_train.loc[ application_train.TARGET == 1].sample( frac=0.1, replace=False )
print( 'label 1 sample size: ', str( application_sample1.shape[0] ) )
application_sample0 = application_train.loc[ application_train.TARGET==0 ].sample( frac=0.1, replace=False )
print( 'label sample size:', str( application_sample0.shape[0] ) )
application = pd.concat( [ application_sample1, application_sample0 ], axis=0 ).sort_values( 'SK_ID_CURR' )

#Impute missing values
categorical_list = []
numerical_list = []
for i in application.columns.tolist():
    if application[i].dtype == 'object':
        categorical_list.append( i )
    else:
        numerical_list.append( i )

print( 'Number of categorical features:', str( len( categorical_list ) ) )
print( 'Number of numberical features:', str( len( numerical_list ) ) )

#Deal with numerical features: median
from sklearn.preprocessing import Imputer
application[ numerical_list ] = Imputer( strategy='median').fit_transform( application[numerical_list] )

#Deal with Categorical features: OneHotEncoding
#del application_train
gc.collect()
application = pd.get_dummies( application, drop_first=True )
print( application.shape )

#Feature matrix and target
X = application.drop( ['SK_ID_CURR', 'TARGET'], axis=1 )
y = application.TARGET
feature_name = X.columns.tolist()

#Feature Selection
#Pearson Correlation
def cor_selector( X, y ):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef( X[i], y )[ 0, 1 ]
        cor_list.append( cor )
    # replace NaN with 0
    cor_list = [ 0 if np.isnan(i) else i for i in cor_list ]
    # feature name
    cor_feature = X.iloc[ :, np.argsort( np.abs( cor_list))[-100:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [ True if i in cor_feature else False for i in feature_name ]
    return cor_support, cor_feature

cor_support, cor_feature = cor_selector( X, y )
print( str( len( cor_feature)), ' selected feature by calculate the correlation:', cor_feature[:20] )

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform( X )
chi_selector = SelectKBest( chi2, k=100 )
chi_selector.fit( X_norm, y )
chi_support = chi_selector.get_support()
chi_feature = X.loc[:, chi_support].columns.tolist()
print( str( len( chi_feature)), " selected feature by chi2: ", chi_feature[:20] )

#Wrapper
from sklearn.feature_selection import  RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE( estimator=LogisticRegression(), n_features_to_select=20, step=10, verbose=5 )
rfe_selector.fit( X_norm, y )
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[ :, rfe_support ].columns.tolist()
print( str( len( rfe_feature ) ), ' selected features by REF: ', rfe_feature[:20] )

#Embeded
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
embeded_lr_selector = SelectFromModel( LogisticRegression( penalty='l1'), '1.25*median' )
embeded_lr_selector.fit( X_norm, y )
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[ :, embeded_lr_support ].columns.tolist()
print( str( len( embeded_lr_feature )), ' selected features by embeded: ', embeded_lr_feature[:20] )

#Random Forest
from sklearn.ensemble import RandomForestClassifier
embeded_rf_selector = SelectFromModel( RandomForestClassifier( n_estimators=100), threshold='1.25*median' )
embeded_rf_selector.fit( X_norm, y )
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[ :, embeded_rf_support ].columns.tolist()
print( str( len( embeded_rf_feature )), ' selected features by Random Forest ', embeded_rf_feature[:20] )

#LightGBM
from lightgbm import  LGBMClassifier
lgbc = LGBMClassifier( n_estimators=500, learning_rate=0.05, num_leaves=256,
                       colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01,
                       min_child_weight=800 )
embeded_lgb_selector = SelectFromModel( lgbc, threshold='1.25*median' )
embeded_lgb_selector.fit( X, y )
embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[ :, embeded_lgb_support ].columns.tolist()
print( str( len( embeded_lgb_feature )), ' selected feautes by LightGBM: ', embeded_lgb_feature[ :20 ])

#Summary
pd.set_option( 'display.max_rows', None )
feature_selection_df = pd.DataFrame( { 'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                       'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support })
#count the selected times for each feature
feature_selection_df['Total'] = np.sum( feature_selection_df, axis=1 )
# display the top 100
feature_selection_df = feature_selection_df.sort_values( ['Total', 'Feature'], ascending=False )
feature_selection_df.index = range( 1, len( feature_selection_df ) + 1 )
print( '---------Summary---------')
print( feature_selection_df[['Feature', 'Total']].head( 100 ) )

X = application[ feature_selection_df.Feature.head( 100) ]