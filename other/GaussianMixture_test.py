'''
@Project: home-credit-default-risk
@Package 
@author: ly
@date Date: 2019年04月15日 13:59
@Description: 
@URL: http://blog.sina.com.cn/s/blog_69e75efd0102wylw.html
@version: V1.0
'''
import numpy as np
from sklearn import mixture

#生成随机观测点，含有2个聚集核心
obs = np.concatenate( ( np.random.randn( 100, 1), 10 + np.random.randn(300,1) ) )
clf = mixture.GaussianMixture( n_components=2 )
print( obs[:10] )

clf.fit( obs )
print( clf.predict(  [ [0], [2], [9], [10]] ) )