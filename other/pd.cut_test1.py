'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月22日 11:41
@Description: 
@URL: https://blog.csdn.net/missyougoon/article/details/83986511
@version: V1.0
'''
import  numpy as np
import pandas as pd
from pandas import Series, DataFrame

np.random.seed( 5643 )

score_list = np.random.randint( 25, 100, size=20 )
print( 'score_list: ', score_list )

bins = [ 0, 59, 70, 80, 100 ]
score_cut = pd.cut( score_list, bins )
#print( type( score_cut ) )
print( '----output-----')
print( score_cut )
#print(pd.value_counts(score_cut)) # 统计每个区间人数

df = DataFrame()
df[ 'score' ] = score_list
df[ 'student' ] = [ pd.util.testing.rands(3) for i in range( len( score_list ) ) ]
print( df )
#print( pd.cut( df['score'], bins ) )
df['categories'] = pd.cut( df['score'], bins )
print( df )

df[ 'categories_lever'] = pd.cut( df['score'], bins, labels=[ 1, 2, 3, 4 ] )
print( df )