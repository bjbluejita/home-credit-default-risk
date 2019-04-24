'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月09日 10:18
@Description: 
@URL: https://github.com/oskird/Kaggle-Home-Credit-Default-Risk-Solution/blob/master/Feature/feature%20-%20installments_payments.py
@version: V1.0
'''
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

installment_payments = pd.read_csv(  '../input/installments_payments.csv' )
installment_payments = installment_payments.sort_values( [ 'SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'] )
previous_application = pd.read_csv( '../input/previous_application.csv' )
previous_application = previous_application.sort_values( [ 'SK_ID_CURR', 'SK_ID_PREV' ] )

#recent application
recent_record = installment_payments.merge( installment_payments.groupby('SK_ID_CURR')['SK_ID_PREV'].max().reset_index().drop( 'SK_ID_CURR', axis=1), on='SK_ID_PREV' )
ip_recent_term = recent_record.groupby( 'SK_ID_CURR' )['NUM_INSTALMENT_NUMBER'].max()
ip_recent_total_actual_payment = recent_record.groupby( 'SK_ID_CURR' )['AMT_PAYMENT'].sum()
ip_recent_total_required_payment = recent_record.groupby( 'SK_ID_CURR')[ 'AMT_INSTALMENT' ].sum()

#recent late & less time
recent_record[ 'is_late' ] = ( recent_record.DAYS_INSTALMENT < recent_record.DAYS_ENTRY_PAYMENT ).map( lambda x: 1 if x==True else 0 ).fillna( 0 )
recent_record[ 'is_less' ] = ( recent_record.AMT_INSTALMENT < recent_record.AMT_PAYMENT ).map( lambda x: 1 if x==True else 0 ).fillna( 0 )
ip_recent_total_late_times = recent_record.groupby( 'SK_ID_CURR' )['is_late'].sum()
ip_recent_total_less_times = recent_record.groupby( 'SK_ID_CURR' )[ 'is_less'].sum()

# recent late & less amount
ip_temp1 = recent_record.loc[ recent_record.is_late==1 ]
ip_temp2 = recent_record.loc[ recent_record.is_less==1 ]
ip_temp1[ 'total_late' ] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_temp1.DAYS_INSTALMENT
ip_temp2[ 'total_less' ] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_recent_total_late_days = ip_temp1.groupby( 'SK_ID_CURR' )['total_late'].sum()
ip_recent_total_less_amount = ip_temp2.groupby( 'SK_ID_CURR' )['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()

#previous application times
ip_prev_count = installment_payments.groupby( 'SK_ID_CURR' )['SK_ID_PREV'].nunique()
ip_payment_count = installment_payments.groupby( 'SK_ID_CURR' )['SK_ID_PREV'].count()

#credit card
installment_payments['IS_CREDIT'] = installment_payments.NUM_INSTALMENT_VERSION.map( lambda x: 1 if x==True else 0 )
ip_creditcard_user = installment_payments.groupby( 'SK_ID_CURR' )['IS_CREDIT'].sum().map( lambda x: 1 if x>0 else 0 )
ip_creditcard_count = installment_payments.groupby( [ 'SK_ID_CURR', 'SK_ID_PREV' ] )['IS_CREDIT'].sum().map( lambda x:1 if x>0 else 0 ).reset_index().groupby('SK_ID_CURR')['IS_CREDIT'].sum()

# change times
ip_temp = ( installment_payments.groupby( ['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique() - 1 ).reset_index()
ip_total_change_times = ip_temp.groupby( 'SK_ID_CURR')['NUM_INSTALMENT_VERSION'].sum()
ip_avg_change_times = ip_temp.groupby( 'SK_ID_CURR')['NUM_INSTALMENT_VERSION'].mean()

# avg instl
ip_temp = installment_payments.groupby( ['SK_ID_CURR', 'SK_ID_PREV'] )['NUM_INSTALMENT_NUMBER'].max().reset_index()
ip_avg_instl = ip_temp.groupby( 'SK_ID_CURR' )['NUM_INSTALMENT_NUMBER'].mean()
ip_max_instl = ip_temp.groupby( 'SK_ID_CURR' )['NUM_INSTALMENT_NUMBER'].max()

# total late & less time
installment_payments['is_late'] = ( installment_payments.DAYS_INSTALMENT < installment_payments.DAYS_ENTRY_PAYMENT ).map( lambda x: 1 if x==True else 0 )
installment_payments['is_less'] = ( installment_payments.AMT_INSTALMENT > installment_payments.AMT_PAYMENT ).map( lambda x: 1 if x==True else 0 )
ip_total_late_times = installment_payments.groupby('SK_ID_CURR')['is_late'].sum()
ip_total_less_times = installment_payments.groupby('SK_ID_CURR')['is_less'].sum()

# total late & less amount
ip_temp1 = installment_payments.loc[ installment_payments.is_late == 1 ]
ip_temp2 = installment_payments.loc[ installment_payments.is_less == 1 ]
ip_temp1['total_late'] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_temp1.DAYS_INSTALMENT
ip_temp2['total_less'] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_total_late_days = ip_temp1.groupby( 'SK_ID_CURR' )['total_late'].sum()
ip_total_less_amount = ip_temp2.groupby('SK_ID_CURR')['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()

# total payment
ip_total_actual_payment = installment_payments.groupby( 'SK_ID_CURR' )['AMT_PAYMENT'].sum()
ip_total_required_payment = installment_payments.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum()

# total late & less time in recent 1 year
ip_1y = installment_payments.loc[ installment_payments.DAYS_ENTRY_PAYMENT > -365 ]
# payment 1 year
ip_payment_count_1y = ip_1y.groupby( 'SK_ID_CURR')['SK_ID_PREV'].count()
ip_creditcard_count_1y = ip_1y.groupby( ['SK_ID_CURR', 'SK_ID_PREV'] )['IS_CREDIT'].sum().map( lambda x:1 if x>0 else 0 ).reset_index().groupby( 'SK_ID_CURR' )['IS_CREDIT'].sum()
ip_total_late_times_1y = ip_1y.groupby( 'SK_ID_CURR' )['is_late'].sum()
ip_total_less_times_1y = ip_1y.groupby(  'SK_ID_CURR' )['is_less'].sum()

# avg instl
ip_temp = ip_1y.groupby( ['SK_ID_CURR', 'SK_ID_PREV'] )['NUM_INSTALMENT_NUMBER'].max().reset_index()
ip_avg_instl_1y = ip_temp.groupby( 'SK_ID_CURR' )['NUM_INSTALMENT_NUMBER'].mean()
# total late & less amount
ip_temp1 = ip_1y.loc[ ip_1y.is_late==1 ]
ip_temp2 = ip_1y.loc[ ip_1y.is_less==1 ]
ip_temp1['total_late'] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_1y.DAYS_INSTALMENT
ip_temp2['total_less'] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_total_late_days_1y = ip_temp1.groupby('SK_ID_CURR')['total_late'].sum()
ip_total_less_amount_1y = ip_temp2.groupby( 'SK_ID_CURR' )['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()
# total payment
ip_total_actual_payment_1y = ip_1y.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum()
ip_total_required_payment_1y = ip_1y.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum()

#total late & less time in recent 6 months
ip_6m = installment_payments.loc[ installment_payments.DAYS_ENTRY_PAYMENT > -180 ]
ip_payment_count_6m = ip_6m.groupby( 'SK_ID_CURR' )['SK_ID_PREV'].count()
ip_creditcard_count_6m = ip_6m.groupby( ['SK_ID_CURR', 'SK_ID_PREV'] )['IS_CREDIT'].sum().map( lambda x: 1 if x>0 else 0 ).reset_index().groupby('SK_ID_CURR')['IS_CREDIT'].sum()
ip_total_late_times_6m = ip_6m.groupby( 'SK_ID_CURR' )['is_late'].sum()
ip_total_less_times_6m = ip_6m.groupby( 'SK_ID_CURR' )['is_less'].sum()
# avg instl
ip_temp = ip_6m.groupby( ['SK_ID_CURR', 'SK_ID_PREV'] )['NUM_INSTALMENT_NUMBER'].max().reset_index()
ip_avg_instl_6m = ip_temp.groupby( 'SK_ID_CURR' )['NUM_INSTALMENT_NUMBER'].mean()
# total late & less amount
ip_temp1 = ip_6m.loc[ ip_6m.is_late==1 ]
ip_temp2 = ip_6m.loc[ ip_6m.is_less==1 ]
ip_temp1['total_late'] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_temp1.DAYS_INSTALMENT
ip_temp2['total_less'] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_total_late_days_6m = ip_temp1.groupby( 'SK_ID_CURR' )['total_late'].sum()
ip_total_less_amount_6m = ip_temp2.groupby( 'SK_ID_CURR' )['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()
# total payment
ip_total_actual_payment_6m = ip_6m.groupby( 'SK_ID_CURR' )['AMT_PAYMENT'].sum()
ip_total_required_payment_6m = ip_6m.groupby( 'SK_ID_CURR' )['AMT_INSTALMENT'].sum()

ip_feature = pd.DataFrame({'ip_prev_count':ip_prev_count, 'ip_payment_count':ip_payment_count, 'ip_creditcard_user':ip_creditcard_user, 'ip_creditcard_count':ip_creditcard_count,
                           'ip_total_change_times':ip_total_change_times, 'ip_avg_change_times':ip_avg_change_times, 'ip_avg_instl':ip_avg_instl, 'ip_max_instl':ip_max_instl,
                           'ip_total_late_times':ip_total_late_times, 'ip_total_less_times':ip_total_less_times, 'ip_total_late_days':ip_total_late_days, 'ip_total_less_amount':ip_total_less_amount,
                           'ip_total_actual_payment':ip_total_actual_payment, 'ip_total_required_payment':ip_total_required_payment,
                           'ip_recent_term':ip_recent_term, 'ip_recent_total_actual_payment':ip_recent_total_actual_payment, 'ip_recent_total_required_payment':ip_recent_total_required_payment,
                           'ip_recent_total_late_times':ip_recent_total_late_times, 'ip_recent_total_less_times':ip_recent_total_less_times,
                           'ip_recent_total_late_days':ip_recent_total_late_days, 'ip_recent_total_less_amount':ip_recent_total_less_amount}).fillna(0)
ip_1y = pd.DataFrame({'ip_payment_count_1y':ip_payment_count_1y, 'ip_creditcard_count_1y': ip_creditcard_count_1y, 'ip_avg_instl_1y':ip_avg_instl_1y,
                      'ip_total_late_times_1y':ip_total_late_times_1y, 'ip_total_less_times_1y':ip_total_less_times_1y, 'ip_total_late_days_1y':ip_total_late_days_1y,
                      'ip_total_less_amount_1y':ip_total_less_amount_1y, 'ip_total_actual_payment_1y':ip_total_actual_payment_1y, 'ip_total_required_payment_1y':ip_total_required_payment_1y}).reset_index().fillna(0)
ip_6m = pd.DataFrame({'ip_payment_count_6m':ip_payment_count_6m, 'ip_creditcard_count_6m': ip_creditcard_count_6m, 'ip_avg_instl_6m':ip_avg_instl_6m,
                      'ip_total_late_times_6m':ip_total_late_times_6m, 'ip_total_less_times_6m':ip_total_less_times_6m, 'ip_total_late_days_6m':ip_total_late_days_6m,
                      'ip_total_less_amount_6m':ip_total_less_amount_6m, 'ip_total_actual_payment_6m':ip_total_actual_payment_6m, 'ip_total_required_payment_6m':ip_total_required_payment_6m}).reset_index().fillna(0)

ip_feature = ip_feature.merge( ip_1y, on='SK_ID_CURR', how='left' ).merge( ip_6m, on='SK_ID_CURR', how='left' )
ip_feature['ip_active_1y'] = ip_feature.ip_total_late_times_1y.notnull().map( lambda x:1 if x==True else 0 )
ip_feature['ip_active_6m'] = ip_feature.ip_total_late_times_6m.notnull().map( lambda x:1 if x==True else 0 )

# active account
ACCOUNT_1Y = installment_payments.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].max().map( lambda x: 1 if x>=-365 else 0 )
ACCOUNT_6M = installment_payments.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].max().map( lambda x: 1 if x>=-180 else 0 )
ACTIVE_ACCOUNT = pd.DataFrame( { 'ACCOUNT_1Y':ACCOUNT_1Y, 'ACCOUNT_6M':ACCOUNT_6M } )
pa_ip = previous_application[ ['SK_ID_CURR', 'SK_ID_PREV','NAME_CONTRACT_TYPE']].merge( ACTIVE_ACCOUNT, on='SK_ID_PREV', how='left' )
COUNT_1Y = pd.get_dummies( pa_ip.loc[ pa_ip.ACCOUNT_1Y==1, ['SK_ID_CURR', 'NAME_CONTRACT_TYPE']], prefix='ip_count_1y_').groupby( 'SK_ID_CURR' ).sum().reset_index()
COUNT_6M = pd.get_dummies( pa_ip.loc[ pa_ip.ACCOUNT_6M==1, ['SK_ID_CURR', 'NAME_CONTRACT_TYPE']], prefix='ip_count_6m_').groupby( 'SK_ID_CURR' ).sum().reset_index()
ip_feature = ip_feature.merge( COUNT_1Y, on='SK_ID_CURR' ).merge( COUNT_6M, on='SK_ID_CURR' )
for i in range( -6, 0 ):
    ip_feature.iloc[ :, i ] = ip_feature.iloc[ :, i ].fillna(0)
print( 'ip_feature shape: ', ip_feature.shape )

ip_feature.to_csv( 'ip_feature.csv', index=False )

