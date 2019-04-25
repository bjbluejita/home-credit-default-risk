'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月24日 12:10
@Description: 
@URL: https://github.com/oskird/Kaggle-Home-Credit-Default-Risk-Solution/blob/master/EDA.ipynb
@version: V1.0
'''
import pandas as pd
import os
import numpy as np
import cmath
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
color = sns.color_palette()
import gc
import warnings
import time
warnings.filterwarnings( 'ignore' )

application_train = pd.read_csv( '../input/application_train.csv' )
application_test = pd.read_csv( '../input/application_test.csv' )
print( 'load finished!')

print( '-------shape-----------' )
print( 'application_train:', application_train.shape )
print( 'application_test:', application_test.shape )

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Plots the disribution of a variable colored by value of the target
def kde_target( var_name, df ):
    # Calculate the correlation coefficient between the new variable and the target
    cor = df['TARGET'].corr( df[var_name] )

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[ df['TARGET']==0, var_name ].median()
    avg_not_repaid = df.ix[ df['TARGET']==1, var_name ].median()

    plt.figure( figsize=(12, 6) )
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot( df.ix[ df['TARGET']==0, var_name ], label='TARGET==0' )
    sns.kdeplot( df.ix[ df['TARGET']==1, var_name ], label='TARGET==1' )

    # label the plot
    plt.xlabel(var_name);plt.ylabel('Density');plt.title( '%s Distribution' % var_name )
    plt.legend()

    # print out the correlation
    print( 'The corelation between %s and the TARGET is %0.4f' %( var_name, cor ))
    # Print out average values
    print( 'Median value for loan was not repaid=%0.4f' % avg_not_repaid )
    print( 'Median value for load was repaid=%0.4f' % avg_repaid )


##清理数据
def clean_data( application, **kwargs ):
    application['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
    application['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    application['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    application['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    application['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
    # application[cfg.CATEGORICAL_COLUMNS].fillna(-1, inplace=True)
    return application

application_train = clean_data( application_train )
application_test = clean_data( application_test )


#A further exploration on application table（进一步探索）
#### Impute missing values（插补缺失值）
from sklearn.impute import SimpleImputer

#split categorical, discrete and numerical features
def feature_type_split( data, special_list=[] ):
    cat_list = []
    dis_num_list = []
    num_list = []

    for i in data.columns.tolist():
        if data[i].dtype == 'object':
            cat_list.append(i)
        elif data[i].nunique() < 25:
            dis_num_list.append(i)
        elif i in special_list:
            dis_num_list.append(i)
        else:
            num_list.append(i)

    return cat_list, dis_num_list, num_list

cat_list, dis_num_list, num_list = feature_type_split( data=application_train,
                                                       special_list=['AMT_REQ_CREDIT_BUREAU_YEAR'])

# discrete
#耗时耗力
print( 'begin impute values' )
start = time.time()
application_train[cat_list] = SimpleImputer(strategy='most_frequent') \
    .fit_transform(application_train[cat_list])
application_test[cat_list]  = SimpleImputer(strategy='most_frequent') \
    .fit_transform(application_test[cat_list])
application_train[dis_num_list]  = SimpleImputer(strategy='most_frequent') \
    .fit_transform(application_train[dis_num_list])
application_test['TARGET'] = ''
application_test[dis_num_list]  = SimpleImputer(strategy='most_frequent') \
    .fit_transform(application_test[dis_num_list])
application_train[num_list] = SimpleImputer(strategy='mean') \
    .fit_transform(application_train[num_list])
application_train[num_list] = SimpleImputer(strategy='most_frequent') \
    .fit_transform(application_train[num_list])
application_test[num_list] = SimpleImputer(strategy='most_frequent') \
    .fit_transform(application_test[num_list])

end = time.time()
print( 'Impute values finished! It takes %d second' % ( end-start ) )

#Create more features
def applicationFeatures( application, **kwargs ):

    #Term: total credit / annuity(贷款总额和年金比是一个非常出色的特征，有很强的预测效果）
    application['TERM'] = application.AMT_CREDIT / application.AMT_ANNUITY

    #OVER_EXPECT_CREDIT: actual credit larger than goods price
    application['OVER_EXPECT_CREDIT'] = ( application.AMT_CREDIT > application.AMT_GOODS_PRICE ).map({False:0, True:1})


    #MEAN_BUILDING_SCORE_TOTAL: the sum of all building AVG score
    application['MEAN_BUILDING_SCORE_AVG'] = application.iloc[:, 44:58 ].mean( skipna=True, axis=1 )

    application['TOTAL_BUILDING_SCORE_AVG'] = application.iloc[:, 44:58].sum( skipna=True, axis=1 )

    #the total number of provided document
    application['FLAG_DOCUMNET_TOTAL'] = application.iloc[:, 96:116].sum( axis=1 )

    #the total number of enquiries
    application['AMT_REQ_CREDIT_BUREAU_TOTAL'] = application.iloc[:, 116:122].sum( axis=1 )

    ##### DAYS_EMPLOYED 异常值处理
    # 为异常值列添加一个新列
    application['DAYS_EMPLOYED_ANOM'] =( application["DAYS_EMPLOYED"] == 365243 ).map( {False:0, True:1})
    # 将异常值替换为中值
    DAYS_EMPLOYED_MEDIAN = application['DAYS_EMPLOYED'].median()
    application['DAYS_EMPLOYED'].replace({365243: DAYS_EMPLOYED_MEDIAN}, inplace = True)

    ##### the days between born and employed
    application['BIRTH_EMPLOTED_INTERVEL'] = application.DAYS_EMPLOYED - application.DAYS_BIRTH

    application['BIRTH_REGISTRATION_INTERVEL'] = application.DAYS_REGISTRATION - application.DAYS_BIRTH

    #####  Building
    application['MEAN_BUILDING_SCORE_AVG'] = application.iloc[:, 44:58 ].mean( skipna=True, axis=1)
    application['TOTAL_BUILDING_SCORE_AVG'] = application.iloc[ :, 44:58 ].sum( axis=1 )

    #### 家庭人均收入
    application['INCOME_PER_FAMILY_MEMBER'] = application.AMT_INCOME_TOTAL / application.CNT_FAM_MEMBERS

    application['SEASON_REMAINING'] = application.AMT_INCOME_TOTAL / 4 - application.AMT_ANNUITY

    ##### AMT_GOODS_PRICE:消费贷款，希望通过贷款购买货物的价格。和收入组合，一定程度表示贷款压力。。。
    application['RATIO_INCOME_GOODS'] = application.AMT_INCOME_TOTAL / application.AMT_GOODS_PRICE

    application['RATIO_INCOME_CREDIT'] = application.AMT_INCOME_TOTAL / application.AMT_CREDIT

    #### log_EXT_SOURCE_1 log_EXT_SOURCE_2 log_EXT_SOURCE_3 是三个非常出色的特征，增加log变换试试
    application['log_EXT_SOURCE_1'] = np.log( application.EXT_SOURCE_1 )
    application['log_EXT_SOURCE_2'] = np.log( application.EXT_SOURCE_2 )
    application['log_EXT_SOURCE_3'] = np.log( application.EXT_SOURCE_3 )

    application['CHILDREN_RATIO'] = application.CNT_CHILDREN / application.CNT_FAM_MEMBERS

    #### 客户年龄异常值处理\客户年龄分段处理
    application['DAYS_BIRTH_YEAR'] =pd.cut( np.abs( np.round( application.DAYS_BIRTH / 365 ) ),  bins = np.linspace(20, 70, num = 11), \
                                            labels=np.linspace( 1, 10, num=10) )
    application['DAYS_BIRTH_log'] = np.log( application.DAYS_BIRTH )

    application['log_TERM'] = np.log( application.TERM )
    application['RATIO_ANNUITY_INCOME'] =  application.AMT_ANNUITY / application.AMT_INCOME_TOTAL
    application['NAME_TYPE_SUITE_LEVEL'] = application['NAME_TYPE_SUITE'].map(lambda x:0 if x=='Unaccompanied' else 1 )
    application['NAME_INCOME_TYPE_LEVEL'] = application.NAME_INCOME_TYPE.map( {'Student': 0, 'Businessman':0,
                                                                               'Pensioner': 1,
                                                                               'State servant': 2,
                                                                               'Commercial associate': 3,
                                                                               'Working': 4,
                                                                               'Unemployed': 5,
                                                                               'Maternity leave': 6} )

    #more application features
    #from URL:https://github.com/neptune-ml/open-solution-home-credit/blob/master/src/feature_extraction.py
    application['annuity_income_percentage'] = application.AMT_ANNUITY / application.AMT_INCOME_TOTAL
    application['cat_to_birth_ratio'] = application.OWN_CAR_AGE / application.DAYS_BIRTH
    application['car_to_employ_ratio'] = application.OWN_CAR_AGE / application.DAYS_EMPLOYED
    application['children_ratio'] = application.CNT_CHILDREN / application.CNT_FAM_MEMBERS
    application['credit_to_annuity_ratio'] = application.AMT_CREDIT / application.AMT_ANNUITY
    application['credit_to_goods_ratio'] = application.AMT_CREDIT / application.AMT_GOODS_PRICE
    application['credit_to_income_ratio'] = application.AMT_CREDIT / application.AMT_INCOME_TOTAL
    application['days_employed_percentage'] = application.AMT_INCOME_TOTAL / application.AMT_CREDIT
    application['income_per_child'] = application.AMT_INCOME_TOTAL / ( 1 + application.CNT_CHILDREN )
    application['income_per_person'] = application.AMT_INCOME_TOTAL / application.CNT_FAM_MEMBERS
    application['payment_rate'] = application.AMT_ANNUITY / application.AMT_CREDIT
    application['phone_to_birth_ratio'] = application.DAYS_LAST_PHONE_CHANGE / application.DAYS_BIRTH
    application['phone_to_employ_ratio'] = application.DAYS_LAST_PHONE_CHANGE / application.DAYS_EMPLOYED
    application['external_sources_weighted'] = np.nansum(
        np.asarray( [ 1.9, 2.1, 2.6] ) * application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1
    )
    application['cnt_non_child'] = application.CNT_FAM_MEMBERS - application.CNT_CHILDREN
    application['child_to_non_child_ratio'] = application.CNT_CHILDREN / application.cnt_non_child
    application['income_per_non_child'] = application.AMT_INCOME_TOTAL / application.cnt_non_child
    application['credit_per_person'] = application.AMT_CREDIT / application.CNT_FAM_MEMBERS
    application['credit_per_child'] = application.AMT_CREDIT / ( 1 + application.CNT_CHILDREN )
    application['credit_per_non_child'] = application.AMT_CREDIT / application.cnt_non_child
    application['ext_source_1_plus_2'] = np.nansum( application[['EXT_SOURCE_1', 'EXT_SOURCE_2']], axis=1 )
    application['ext_source_1_plus_3'] = np.nansum( application[['EXT_SOURCE_1', 'EXT_SOURCE_3']], axis=1 )
    application['ext_source_2_plus_3'] = np.nansum( application[['EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1 )
    application['ext_source_1_is_nan'] = np.isnan( application.EXT_SOURCE_1 ).map( {False:1, True:0} )
    application['ext_source_2_is_nan'] = np.isnan( application.EXT_SOURCE_2 ).map( {False:1, True:0} )
    application['ext_source_3_is_nan'] = np.isnan( application.EXT_SOURCE_3 ).map( {False:1, True:0} )
    application['hour_appr_process_start_radial_x'] = application.HOUR_APPR_PROCESS_START.apply(
        lambda x: cmath.rect( 1, 2 * cmath.pi * x / 24 ).real
    )
    application['hour_appr_process_start_radial_y'] = application.HOUR_APPR_PROCESS_START.apply(
        lambda x: cmath.rect( 1, 2 * cmath.pi * x / 24 ).imag
    )
    application['id_renewal_days'] = application.DAYS_ID_PUBLISH - application.DAYS_BIRTH
    application['id_renewal_years'] = ( application.DAYS_ID_PUBLISH - application.DAYS_BIRTH ) / 365
    application['id_renewal_days_issue'] = np.vectorize(
        lambda x: max( list( set( [ min( x, age) for age in [0, 20*365, 25*365, 45*365]]) - set([x])))) \
        ( application.id_renewal_days )
    application['id_renewal_years_issue'] = np.vectorize(
        lambda x: max( list( set( [ min( x, age) for age in [0, 20, 25, 46]]) - set([x])))) \
        ( application.id_renewal_years )
    application.loc[ application['id_renewal_days_issue'] <= 20*366, 'id_renewal_days_delay' ] = -1
    application.loc[ application['id_renewal_years_issue'] <= 20*366, 'id_renewal_years_delay' ] = -1
    application.loc[ application['id_renewal_days_issue'] > 20*366, 'id_renewal_days_delay' ] = \
        application.loc[ application['id_renewal_days_issue'] > 20*366, 'id_renewal_days' ].values - \
        application.loc[ application['id_renewal_days_issue'] > 20*366, 'id_renewal_days_issue' ]
    application.loc[ application['id_renewal_years_issue']>20, 'id_renewal_years_delay' ] = \
        application.loc[ application['id_renewal_years_issue']>20, 'id_renewal_years'].values - \
        application.loc[ application['id_renewal_years_issue']>20, 'id_renewal_years_issue' ]

    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
        application[  'external_source_{}'.format( function_name ) ] = eval( 'np.{}'.format( function_name ))(
            application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1
        )

    application['short_employment'] = ( application.DAYS_EMPLOYED < -2000 ).astype( int )
    application['young_age'] = ( application.DAYS_BIRTH < -14000 ).astype( int )\

    return application

application_train = applicationFeatures( application_train )
application_test = applicationFeatures( application_test )


#####  convert categorical variables to numericals
from sklearn.preprocessing import LabelEncoder
def label_encoder( input_df, encoder_dict=None ):
    #Process a dataframe into a form useable by LightGBM
    #Label encoder categoricals
    categorical_feats = input_df.columns[ input_df.dtypes=='object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform( input_df[feat].fillna('NULL'))

    return input_df, categorical_feats.tolist(), encoder_dict

#调用
application_train, categorical_feats, encoder_dict = label_encoder( application_train )
application_test, categorical_feats, encoder_dict = label_encoder( application_test )

#### 保存结果
application_train.to_csv( 'application_train_processed.csv')
application_test.to_csv( 'application_test_processed.csv' )
print( 'saved!' )