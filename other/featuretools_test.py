'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月17日 14:49
@Description: 
@URL: https://github.com/scottlinlin/auto_feature_demo
@version: V1.0
'''
import pandas as pd
import featuretools as ft
import numpy as np

#load data
clients = pd.read_csv( './featuretools_data/clients.csv', parse_dates=['joined'] )
loan = pd.read_csv( './featuretools_data/loans.csv', parse_dates=['loan_start', 'loan_end'] )
payments = pd.read_csv( './featuretools_data/payments.csv', parse_dates=['payment_date'] )

#创建实体和实体集
es = ft.EntitySet( id='clients' )

#添加clients实体
es = es.entity_from_dataframe( entity_id='clients', dataframe=clients,
                               index='client_id', time_index='joined' )

#添加loads实体
es = es.entity_from_dataframe( entity_id='loans', dataframe=loan,
                               variable_types={'repaid': ft.variable_types.Categorical},
                               index='loan_id', time_index='loan_start' )

#添加pyments实体
es = es.entity_from_dataframe( entity_id='payments',
                               dataframe=payments,
                               variable_types={'missed': ft.variable_types.Categorical},
                               make_index=True,
                               index='payment_id',
                               time_index='payment_date')
print( es )
#添加实体关系
r_client_previous = ft.Relationship( es['clients']['client_id'], es['loans']['client_id'] )
es = es.add_relationship( r_client_previous )

r_payments = ft.Relationship( es['loans']['loan_id'], es['payments']['loan_id'] )
es = es.add_relationship( r_payments)

print( es )
#聚合特征,并生成新特征
features, feature_names = ft.dfs( entityset=es, target_entity='clients',
                                  verbose=True )
features.to_csv( './featuretools_data/features_new_1.csv', index=False )

#聚合特征，通过指定聚合和转换函数生成新特征
features, feature_names = ft.dfs( entityset=es, target_entity='clients',
                                  agg_primitives=['mean', 'max', 'percent_true', 'last', 'trend'],
                                  trans_primitives=['year', 'month', ], verbose=True )
features.to_csv( './featuretools_data/features_new_2.csv', index=False )