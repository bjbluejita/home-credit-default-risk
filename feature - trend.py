'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月15日 15:38
@Description: 
@URL: https://github.com/oskird/Kaggle-Home-Credit-Default-Risk-Solution/blob/master/Feature/feature%20-%20trend.py
@version: V1.0
'''
import numpy as np
import pandas as pd
import featuretools as ft
import gc
import warnings
import time
start = time.time()
row1 = None
row2 = None
row3 = None
print( 'begin load data')
app_train = pd.read_csv( './data/application_train.csv', nrows=row1 ).sort_values( 'SK_ID_CURR' )
app_test = pd.read_csv( './data/application_test.csv', nrows=row1 ).sort_values( 'SK_ID_CURR' )
bureau   = pd.read_csv( './data/bureau.csv', nrows=row2 ).sort_values( ['SK_ID_CURR', 'SK_ID_BUREAU'] )
bureau_balance = pd.read_csv( './data/bureau_balance.csv', nrows=row3 ).sort_values( ['SK_ID_BUREAU', 'MONTHS_BALANCE'] )
previous = pd.read_csv( './data/previous_application.csv', nrows=row3 ).sort_values( ['SK_ID_CURR', 'SK_ID_PREV'] )
cash     = pd.read_csv( './data/POS_CASH_balance.csv', nrows=row3 ).sort_values( ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE' ] )
credit   = pd.read_csv( './data/credit_card_balance.csv', nrows=row3 ).sort_values( ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'] )
installments = pd.read_csv( './data/installments_payments.csv', nrows=row3 ).sort_values( ['SK_ID_CURR', 'SK_ID_PREV'] )
print( 'end load data')

#data manipulation
print( 'begin data manipulation')
app_train = app_train[['SK_ID_CURR']]
app_test = app_test[['SK_ID_CURR']]
bureau = bureau[ ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
                  'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE']]
bureau_balance.STATUS = bureau_balance.STATUS.map( { 'C':0, 'X':0.1, '1':1., '2':2., '3':3., '4':4., '5':5. } )
previous = previous[ ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION',
                      'CNT_PAYMENT', 'NAME_YIELD_GROUP', 'NFLAG_INSURED_ON_APPROVAL', 'SELLERPLACE_AREA' ]]
previous.NAME_CONTRACT_STATUS = previous.NAME_CONTRACT_STATUS.map( lambda x: 1 if x=='Refused' else 0 )
previous.NAME_YIELD_GROUP = previous.NAME_YIELD_GROUP.map( { 'XNA':0, 'low_noraml':1, 'low_action':1, 'middle':2, 'high':3 } )
cash = cash[ ['SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF' ]]
installments['date_diff'] = installments.DAYS_INSTALMENT - installments.DAYS_ENTRY_PAYMENT
installments['amount_diff'] = installments.AMT_INSTALMENT - installments.AMT_PAYMENT
installments = installments[ ['SK_ID_PREV', 'DAYS_INSTALMENT', 'amount_diff','date_diff' ] ]

#time
import re

def replace_day_outliers( df ):
    #Replace 365243 with np.nan in any columns with DAYS
    for col in df.columns:
        if 'DAY' in col:
            df[col] = df[col].replace( 365243, np.nan )

    return df

# Replace all the day outliers
app_train = replace_day_outliers( app_train )
app_test = replace_day_outliers( app_test )
bureau = replace_day_outliers( bureau )
bureau_balance = replace_day_outliers( bureau_balance )
previous = replace_day_outliers( previous )
cash = replace_day_outliers( cash )
credit = replace_day_outliers( credit )
installments = replace_day_outliers( installments )

start_date = pd.Timestamp( '2018-01-01' )

# bureau
for col in ['DAYS_CREDIT']:
    bureau[col] = pd.to_timedelta( bureau[col], 'D' )

# Create the date columns
bureau['bureau_credit_application_date'] = start_date + bureau['DAYS_CREDIT']

#balance
bureau_balance['MONTHS_BALANCE'] = pd.to_timedelta( bureau_balance['MONTHS_BALANCE'], 'M' )
# Make a date column
bureau_balance['bureau_balance_date'] = start_date + bureau_balance['MONTHS_BALANCE']
bureau_balance = bureau_balance.drop( columns=['MONTHS_BALANCE'] )


#Convert to timedeltas in days
for col in ['DAYS_DECISION']:
    previous[col] = pd.to_timedelta( previous[col], 'D' )

#make date columns
previous['previous_decision_date'] = start_date + previous['DAYS_DECISION']

#Drop the time offset columns
previous = previous.drop( columns=['DAYS_DECISION'] )

#CASH
cash['MONTHS_BALANCE'] = pd.to_timedelta(cash['MONTHS_BALANCE'], 'M')
cash['cash_balance_date'] = start_date + cash['MONTHS_BALANCE']
cash = cash.drop(columns = ['MONTHS_BALANCE'])

#CREDIT
credit['MONTHS_BALANCE'] = pd.to_timedelta( credit['MONTHS_BALANCE'], 'M' )
credit['credit_balance_date'] = start_date + credit['MONTHS_BALANCE']
credit = credit.drop( columns=['MONTHS_BALANCE'] )

#installment
installments['DAYS_INSTALMENT'] = pd.to_timedelta( installments['DAYS_INSTALMENT'], 'D' )
installments['installments_due_date'] = start_date + installments['DAYS_INSTALMENT']
installments = installments.drop( columns=['DAYS_INSTALMENT'] )

print( 'begin make an antityset' )
# Make an entityset
es = ft.EntitySet( id='clients' )

es = es.entity_from_dataframe( entity_id='app_train', dataframe=app_train,
                               index='SK_ID_CURR' )

es = es.entity_from_dataframe( entity_id='app_test', dataframe=app_test,
                               index='SK_ID_CURR' )

# part 1
es = es.entity_from_dataframe( entity_id='bureau', dataframe=bureau,
                               index='SK_ID_BUREAU', time_index='bureau_credit_application_date' )
es = es.entity_from_dataframe( entity_id='bureau_balance', dataframe=bureau_balance,
                               make_index=True, index='bb_index',
                               time_index='bureau_balance_date' )

# part 2
es = es.entity_from_dataframe( entity_id='previous', dataframe=previous,
                               index='SK_ID_PREV', time_index='previous_decision_date' )
es = es.entity_from_dataframe( entity_id='cash', dataframe=cash,
                               make_index=True, index='cash_index',
                               time_index='cash_balance_date' )
es = es.entity_from_dataframe( entity_id='installments', dataframe=installments,
                               make_index=True, index='installments_index',
                               time_index='installments_due_date' )
es = es.entity_from_dataframe( entity_id='credit', dataframe=credit,
                               make_index=True, index='credit_index',
                               time_index='credit_balance_date' )

# Relationship between app and bureau
r_app_bureau = ft.Relationship( es['app_train']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'] )

# Test Relationship between app and bureau
r_test_app_bureau = ft.Relationship( es['app_test']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'] )

# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship( es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'] )

# Relationship between current app and previous apps
r_app_previous = ft.Relationship( es['app_train']['SK_ID_CURR'], es['previous']['SK_ID_CURR'] )

# Test Relationship between current app and previous apps
r_test_app_previous = ft.Relationship( es['app_test']['SK_ID_CURR'], es['previous']['SK_ID_CURR'] )

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship( es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'] )
r_previous_installments = ft.Relationship( es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'] )
r_previous_credit = ft.Relationship(  es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'] )

# Add in the defined relationships
es = es.add_relationships( [ r_app_bureau, r_test_app_bureau, r_bureau_balance,
                            r_app_previous, r_test_app_previous, r_previous_cash,
                            r_previous_installments, r_previous_credit ])
del app_train, app_test, bureau, bureau_balance, previous, cash, credit,installments
gc.collect()
print( 'prepare time:', time.time() - start )

#train feature
start = time.time()
time_features, time_feature_names = ft.dfs( entityset=es, target_entity='app_train',
                                            max_depth=2, agg_primitives=['trend'],
                                            features_only=False, verbose=True,
                                            chunk_size=25000,
                                            ignore_entities=['app_test'] )
print( 'app_train ft.dfs time: ', time.time() - start )
time_features.reset_index().to_csv( 'trend_train.csv', index=False )

# test features
start = time.time()
time_features_test, time_feature_names = ft.dfs( entityset=es, target_entity='app_test',
                                                 max_depth=2, agg_primitives=['trend'],
                                                 features_only=False, verbose=True,
                                                 chunk_size=25000,
                                                 ignore_entities=['app_train'] )
time_features_test.reset_index().to_csv( 'trennd_test.csv', index=False )
print( 'app_test ft.dfs time: ', time.time() - start )