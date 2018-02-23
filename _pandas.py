import pandas as pd
import numpy as np

s = pd.Series([1,2,34,np.nan,'str'])
'''
0      1
1      2
2     34
3    NaN
4    str
dtype: object
'''

dates = pd.date_range('20180201',periods = 6)
'''
DatetimeIndex(['2018-02-01', '2018-02-02', '2018-02-03', '2018-02-04',
               '2018-02-05', '2018-02-06'],
              dtype='datetime64[ns]', freq='D')
'''
df = pd.DataFrame(np.random.randn(6,4),index = dates,columns = ['a','b','c','d']) 
# np.random.randn(6,4) or np.random.random((6,4)) 注意括号数量 
'''
                   a         b         c         d
2018-02-01  1.391946 -2.424893 -1.232677  0.131264
2018-02-02  1.033156  1.742443 -0.817066  0.241889
2018-02-03 -2.249280 -0.063745  0.114858 -0.557182
2018-02-04 -0.354538  1.306568  0.213278 -0.753626
2018-02-05  0.753679 -1.599087  0.171812 -1.288954
2018-02-06  0.682236 -1.296736 -0.208484 -0.105507
'''
df = pd.DataFrame(np.arange(12).reshape(3,4))
'''
   0  1   2   3
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
'''
df = pd.DataFrame({
	'A':1,
	'B':pd.Timestamp('20180202'),
	'C':pd.Series(1,index = list(range(4)),dtype = 'float32'),
	'D':np.array([3] * 4,dtype = 'int32'),
	'E':pd.Categorical(["test","train","test","train"]),
	'F':'foo'
	})
'''
   A          B    C  D      E    F
0  1 2018-02-02  1.0  3   test  foo
1  1 2018-02-02  1.0  3  train  foo
2  1 2018-02-02  1.0  3   test  foo
3  1 2018-02-02  1.0  3  train  foo
'''
df.dtypes
'''
A             int64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
'''
df.index # Int64Index([0, 1, 2, 3], dtype='int64')
df.columns # Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')
df.values
'''
[[1 Timestamp('2018-02-02 00:00:00') 1.0 3 'test' 'foo']
 [1 Timestamp('2018-02-02 00:00:00') 1.0 3 'train' 'foo']
 [1 Timestamp('2018-02-02 00:00:00') 1.0 3 'test' 'foo']
 [1 Timestamp('2018-02-02 00:00:00') 1.0 3 'train' 'foo']]
'''
df.describe() # 只运算可以运算的
'''
         A    C    D
count  4.0  4.0  4.0
mean   1.0  1.0  3.0
std    0.0  0.0  0.0
min    1.0  1.0  3.0
25%    1.0  1.0  3.0
50%    1.0  1.0  3.0
75%    1.0  1.0  3.0
max    1.0  1.0  3.0
'''
df.T # 转置
df.sort_index(axis = 1,ascending = False)
'''
     F      E  D    C          B  A
0  foo   test  3  1.0 2018-02-02  1
1  foo  train  3  1.0 2018-02-02  1
2  foo   test  3  1.0 2018-02-02  1
3  foo  train  3  1.0 2018-02-02  1
'''
df.sort_values(by = 'E')
'''
   A          B    C  D      E    F
0  1 2018-02-02  1.0  3   test  foo
2  1 2018-02-02  1.0  3   test  foo
1  1 2018-02-02  1.0  3  train  foo
3  1 2018-02-02  1.0  3  train  foo
'''

