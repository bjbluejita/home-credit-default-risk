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
#Term: total credit / annuity(贷款总额和年金比是一个非常出色的特征，有很强的预测效果）
application_train['TERM'] = application_train.AMT_CREDIT / application_train.AMT_ANNUITY
application_test['TERM'] = application_test.AMT_CREDIT / application_test.AMT_ANNUITY

#OVER_EXPECT_CREDIT: actual credit larger than goods price
application_train['OVER_EXPECT_CREDIT'] = ( application_train.AMT_CREDIT > application_train.AMT_GOODS_PRICE ).map({False:0, True:1})
application_test['OVER_EXPECT_CREDIT'] = ( application_test.AMT_CREDIT > application_test.AMT_GOODS_PRICE ).map({False:0, True:1})

#MEAN_BUILDING_SCORE_TOTAL: the sum of all building AVG score
application_train['MEAN_BUILDING_SCORE_AVG'] = application_train.iloc[:, 44:58 ].mean( skipna=True, axis=1 )
application_test['MEAN_BUILDING_SCORE_AVG'] = application_test.iloc[:, 44:58 ].mean( skipna=True, axis=1 )

application_train['TOTAL_BUILDING_SCORE_AVG'] = application_train.iloc[:, 44:58].sum( skipna=True, axis=1 )
application_test['TOTAL_BUILDING_SCORE_AVG'] = application_test.iloc[:, 44:58].sum( skipna=True, axis=1 )

#the total number of provided document
application_train['FLAG_DOCUMNET_TOTAL'] = application_train.iloc[:, 96:116].sum( axis=1 )
application_test['FLAG_DOCUMENT_TOTAL'] = application_test.iloc[:, 96:116].sum( axis=1 )

#the total number of enquiries
application_train['AMT_REQ_CREDIT_BUREAU_TOTAL'] = application_train.iloc[:, 116:122].sum( axis=1 )
application_test['AMT_REQ_CREDIT_BUREAU_TOTAL'] = application_test.iloc[:, 116:122].sum( axis=1 )

##### DAYS_EMPLOYED 异常值处理
# 为异常值列添加一个新列
application_train['DAYS_EMPLOYED_ANOM'] =( application_train["DAYS_EMPLOYED"] == 365243 ).map( {False:0, True:1})
# 将异常值替换为中值
DAYS_EMPLOYED_MEDIAN = application_train['DAYS_EMPLOYED'].median()
application_train['DAYS_EMPLOYED'].replace({365243: DAYS_EMPLOYED_MEDIAN}, inplace = True)

# 为异常值列添加一个新列
application_test['DAYS_EMPLOYED_ANOM'] = (application_test["DAYS_EMPLOYED"] == 365243).map( {False:0, True:1} )
# 将异常值替换为中值
DAYS_EMPLOYED_MEDIAN = application_test['DAYS_EMPLOYED'].median()
application_test['DAYS_EMPLOYED'].replace({365243: DAYS_EMPLOYED_MEDIAN }, inplace = True)

##### the days between born and employed
application_train['BIRTH_EMPLOTED_INTERVEL'] = application_train.DAYS_EMPLOYED - application_train.DAYS_BIRTH
application_test['BIRTH_EMPLOTED_INTERVEL'] = application_test.DAYS_EMPLOYED - application_test.DAYS_BIRTH

application_train['BIRTH_REGISTRATION_INTERVEL'] = application_train.DAYS_REGISTRATION - application_train.DAYS_BIRTH
application_test['BIRTH_REGISTRATION_INTERVEL'] = application_test.DAYS_REGISTRATION - application_test.DAYS_BIRTH

#####  Building
application_train['MEAN_BUILDING_SCORE_AVG'] = application_train.iloc[:, 44:58 ].mean( skipna=True, axis=1)
application_train['TOTAL_BUILDING_SCORE_AVG'] = application_train.iloc[ :, 44:58 ].sum( axis=1 )
application_test['MEAN_BUILDING_SCORE_AVG'] = application_test.iloc[:, 44:58 ].mean( skipna=True, axis=1)
application_test['TOTAL_BUILDING_SCORE_AVG'] = application_test.iloc[ :, 44:58 ].sum( axis=1 )

#### 家庭人均收入
application_train['INCOME_PER_FAMILY_MEMBER'] = application_train.AMT_INCOME_TOTAL / application_train.CNT_FAM_MEMBERS
application_test['INCOME_PER_FAMILY_MEMBER'] = application_test.AMT_INCOME_TOTAL / application_test.CNT_FAM_MEMBERS

application_train['SEASON_REMAINING'] = application_train.AMT_INCOME_TOTAL / 4 - application_train.AMT_ANNUITY
application_test['SEASON_REMAINING'] = application_test.AMT_INCOME_TOTAL / 4 - application_test.AMT_ANNUITY

##### AMT_GOODS_PRICE:消费贷款，希望通过贷款购买货物的价格。和收入组合，一定程度表示贷款压力。。。
application_train['RATIO_INCOME_GOODS'] = application_train.AMT_INCOME_TOTAL / application_train.AMT_GOODS_PRICE
application_test['RATIO_INCOME_GOODS'] = application_test.AMT_INCOME_TOTAL / application_test.AMT_GOODS_PRICE

application_train['RATIO_INCOME_CREDIT'] = application_train.AMT_INCOME_TOTAL / application_train.AMT_CREDIT
application_test['RATIO_INCOME_CREDIT'] = application_test.AMT_INCOME_TOTAL / application_test.AMT_CREDIT

#### log_EXT_SOURCE_1 log_EXT_SOURCE_2 log_EXT_SOURCE_3 是三个非常出色的特征，增加log变换试试
application_train['log_EXT_SOURCE_1'] = np.log( application_train.EXT_SOURCE_1 )
application_train['log_EXT_SOURCE_2'] = np.log( application_train.EXT_SOURCE_2 )
application_train['log_EXT_SOURCE_3'] = np.log( application_train.EXT_SOURCE_3 )

application_test['log_EXT_SOURCE_1'] = np.log( application_test.EXT_SOURCE_1 )
application_test['log_EXT_SOURCE_2'] = np.log( application_test.EXT_SOURCE_2 )
application_test['log_EXT_SOURCE_3'] = np.log( application_test.EXT_SOURCE_3 )

application_train['CHILDREN_RATIO'] = application_train.CNT_CHILDREN / application_train.CNT_FAM_MEMBERS
application_test['CHILDREN_RATIO'] = application_test.CNT_CHILDREN / application_test.CNT_FAM_MEMBERS

#### 客户年龄异常值处理\客户年龄分段处理
application_train['DAYS_BIRTH_YEAR'] =pd.cut( np.abs( np.round( application_train.DAYS_BIRTH / 365 ) ),  bins = np.linspace(20, 70, num = 11), \
                                              labels=np.linspace( 1, 10, num=10) )
application_test['DAYS_BIRTH_YEAR'] = pd.cut( np.abs( np.round( application_test.DAYS_BIRTH / 365 ) ),  bins = np.linspace(20, 70, num = 11), \
                                              labels=np.linspace( 1, 10, num=10) )
application_train['DAYS_BIRTH_log'] = np.log( application_train.DAYS_BIRTH )
application_test['DAYS_BIRTH_log'] = np.log( application_test.DAYS_BIRTH )

def income_type_to_level( income_type ):
    if income_type == 'Student' or income_type == 'Businessman': return 0
    elif income_type == 'Pensioner': return 1
    elif income_type == 'State servant': return 2
    elif income_type == 'Commercial associate': return 3
    elif income_type == 'Working': return 4
    elif income_type == 'Unemployed': return 5
    elif income_type == 'Maternity leave': return 6
    else: return 0

application_train['log_TERM'] = np.log( application_train.TERM )
application_train['RATIO_ANNUITY_INCOME'] =  application_train.AMT_ANNUITY / application_train.AMT_INCOME_TOTAL
application_train['NAME_TYPE_SUITE_LEVEL'] = application_train['NAME_TYPE_SUITE'].map(lambda x:0 if x=='Unaccompanied' else 1 )
application_train['NAME_INCOME_TYPE_LEVEL'] = application_train.NAME_INCOME_TYPE.map( lambda x: income_type_to_level(x) )


application_test['log_TERM'] = np.log( application_test.TERM )
application_test['RATIO_ANNUITY_INCOME'] =  application_test.AMT_ANNUITY / application_test.AMT_INCOME_TOTAL
application_test['NAME_TYPE_SUITE_LEVEL'] = application_test['NAME_TYPE_SUITE'].map(lambda x:0 if x=='Unaccompanied' else 1 )
application_test['NAME_INCOME_TYPE_LEVEL'] = application_test.NAME_INCOME_TYPE.map( lambda x: income_type_to_level(x) )

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