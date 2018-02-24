import pandas as pd
import numpy as np


# 属性----------------------------------------------------------------------------
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


# 数据选择----------------------------------------------------------------------------
dates = pd.date_range('20180201',periods = 6)
df = pd.DataFrame(np.arange(24).reshape(6,4),index = dates,columns = ['A','B','C','D'])
'''
             A   B   C   D
2018-02-01   0   1   2   3
2018-02-02   4   5   6   7
2018-02-03   8   9  10  11
2018-02-04  12  13  14  15
2018-02-05  16  17  18  19
2018-02-06  20  21  22  23
'''
df['A'] # 选择A这一列
df.A # 同上
'''
2018-02-01     0
2018-02-02     4
2018-02-03     8
2018-02-04    12
2018-02-05    16
2018-02-06    20
Freq: D, Name: A, dtype: int64 
'''
df[0:3] # 选择前3行
df['20180201':'20180203'] # 同上
'''
            A  B   C   D
2018-02-01  0  1   2   3
2018-02-02  4  5   6   7
2018-02-03  8  9  10  11
'''
# select by label:loc
df.loc['20180202']
'''
A    4
B    5
C    6
D    7
Name: 2018-02-02 00:00:00, dtype: int64
'''
df.loc['20180202',['A','B']]
'''
A    4
B    5
Name: 2018-02-02 00:00:00, dtype: int64
'''
# select by position:iloc
df.iloc[3:5,1:3]
'''
             B   C
2018-02-04  13  14
2018-02-05  17  18
'''
df.iloc[[1,3,5],1:3]
'''
             B   C
2018-02-02   5   6
2018-02-04  13  14
2018-02-06  21  22
'''
# mixed selection:ix
df.ix[:3,['A','C']]
'''
            A   C
2018-02-01  0   2
2018-02-02  4   6
2018-02-03  8  10
'''
# Boolean indexing
df[df.A > 8]
'''
             A   B   C   D
2018-02-04  12  13  14  15
2018-02-05  16  17  18  19
2018-02-06  20  21  22  23
'''


# 设置值----------------------------------------------------------------------------
'''
             A   B   C   D
2018-02-01   0   1   2   3
2018-02-02   4   5   6   7
2018-02-03   8   9  10  11
2018-02-04  12  13  14  15
2018-02-05  16  17  18  19
2018-02-06  20  21  22  23
'''
df.iloc[2,2] = 111
df.loc['20180202','B'] = 222
df.B[df.A > 4] = 0
df['F'] = np.nan
df['E'] = pd.Series([1,2,3,4,5,6],index = pd.date_range('20180201',periods = 6))
'''
             A    B    C   D   F  E
2018-02-01   0    1    2   3 NaN  1
2018-02-02   4  222    6   7 NaN  2
2018-02-03   8    0  111  11 NaN  3
2018-02-04  12    0   14  15 NaN  4
2018-02-05  16    0   18  19 NaN  5
2018-02-06  20    0   22  23 NaN  6
''' 


# 处理丢失数据----------------------------------------------------------------------------
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
df.dropna(axis = 0,how = 'any') # how = {'any',all}
'''
Empty DataFrame
Columns: [A, B, C, D, F, E]
Index: []
'''
df.dropna(axis = 1,how = 'all')
'''
             A      B      C   D  E
2018-02-01   0    NaN    2.0   3  1
2018-02-02   4  222.0    NaN   7  2
2018-02-03   8    0.0  111.0  11  3
2018-02-04  12    0.0   14.0  15  4
2018-02-05  16    0.0   18.0  19  5
2018-02-06  20    0.0   22.0  23  6
'''
df.fillna(value = 0) # fill nan
'''
             A      B      C   D    F  E
2018-02-01   0    0.0    2.0   3  0.0  1
2018-02-02   4  222.0    0.0   7  0.0  2
2018-02-03   8    0.0  111.0  11  0.0  3
2018-02-04  12    0.0   14.0  15  0.0  4
2018-02-05  16    0.0   18.0  19  0.0  5
2018-02-06  20    0.0   22.0  23  0.0  6
'''
df.isnull()
'''
                A      B      C      D     F      E
2018-02-01  False   True  False  False  True  False
2018-02-02  False  False   True  False  True  False
2018-02-03  False  False  False  False  True  False
2018-02-04  False  False  False  False  True  False
2018-02-05  False  False  False  False  True  False
2018-02-06  False  False  False  False  True  False
'''
np.any(df.isnull()) == True # True


# 数据导入导出----------------------------------------------------------------------------
data = pd.read_csv('student.csv')
data.to_pickle('student.pickle')


# Pandas合并----------------------------------------------------------------------------
# concatenating
df1 = pd.DataFrame(np.ones((3,4)) * 0,columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4)) * 1,columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4)) * 2,columns = ['a','b','c','d'])
res = pd.concat([df1,df2,df3],axis = 0)
'''
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
0  1.0  1.0  1.0  1.0
1  1.0  1.0  1.0  1.0
2  1.0  1.0  1.0  1.0
0  2.0  2.0  2.0  2.0
1  2.0  2.0  2.0  2.0
2  2.0  2.0  2.0  2.0
'''
res = pd.concat([df1,df2,df3],axis = 0,ignore_index = True)
'''
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  1.0  1.0  1.0  1.0
4  1.0  1.0  1.0  1.0
5  1.0  1.0  1.0  1.0
6  2.0  2.0  2.0  2.0
7  2.0  2.0  2.0  2.0
8  2.0  2.0  2.0  2.0
'''

# join{'inner','outer'}
df1 = pd.DataFrame(np.ones((3,4)) * 0,columns = ['a','b','c','d'],index = [1,2,3])
df2 = pd.DataFrame(np.ones((3,4)) * 1,columns = ['b','c','d','e'],index = [2,3,4])
res = pd.concat([df1,df2]) # axis = 0,join = 'outer'
'''
     a    b    c    d    e
1  0.0  0.0  0.0  0.0  NaN
2  0.0  0.0  0.0  0.0  NaN
3  0.0  0.0  0.0  0.0  NaN
2  NaN  1.0  1.0  1.0  1.0
3  NaN  1.0  1.0  1.0  1.0
4  NaN  1.0  1.0  1.0  1.0
'''
res = pd.concat([df1,df2],join = 'inner',ignore_index = True)
'''
     b    c    d
0  0.0  0.0  0.0
1  0.0  0.0  0.0
2  0.0  0.0  0.0
3  1.0  1.0  1.0
4  1.0  1.0  1.0
5  1.0  1.0  1.0
'''

# join_axes
res = pd.concat([df1,df2],axis = 1,join_axes = [df1.index]) # 若没有join_axes 则出现第四行
'''
     a    b    c    d    b    c    d    e
1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
'''

# append 只能竖向加数据!!!!
df1 = pd.DataFrame(np.ones((3,4)) * 0,columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4)) * 1,columns = ['a','b','c','d'])
res = df1.append(df2,ignore_index = True)
'''
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  1.0  1.0  1.0  1.0
4  1.0  1.0  1.0  1.0
5  1.0  1.0  1.0  1.0
'''
res = df1.append([df2,df2],ignore_index = True) # 多个
s1 = pd.Series([1,2,3,4],index = ['a','b','c','d'])
res = df1.append(s1,ignore_index = True)
'''
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  1.0  2.0  3.0  4.0
'''


# Pandas合并----------------------------------------------------------------------------
# merging two df by key/keys.(may be used in database)
left = pd.DataFrame({
	'key': ['K0', 'K1', 'K2', 'K3'],
	'A': ['A0', 'A1', 'A2', 'A3'],
	'B': ['B0', 'B1', 'B2', 'B3']
	})
right = pd.DataFrame({
	'key': ['K0', 'K1', 'K2', 'K3'],
	'C': ['C0', 'C1', 'C2', 'C3'],
	'D': ['D0', 'D1', 'D2', 'D3']
	})
res = pd.merge(left,right,on = 'key')
'''
    A   B key   C   D
0  A0  B0  K0  C0  D0
1  A1  B1  K1  C1  D1
2  A2  B2  K2  C2  D2
3  A3  B3  K3  C3  D3
'''

# consider two keys
left = pd.DataFrame({
	'key1': ['K0', 'K0', 'K1', 'K2'],
	'key2': ['K0', 'K1', 'K0', 'K1'],
	'A': ['A0', 'A1', 'A2', 'A3'],
	'B': ['B0', 'B1', 'B2', 'B3']
	})
right = pd.DataFrame({
	'key1': ['K0', 'K1', 'K1', 'K2'],
	'key2': ['K0', 'K0', 'K0', 'K0'],
	'C': ['C0', 'C1', 'C2', 'C3'],
	'D': ['D0', 'D1', 'D2', 'D3']
	})
'''
    A   B key1 key2
0  A0  B0   K0   K0
1  A1  B1   K0   K1
2  A2  B2   K1   K0
3  A3  B3   K2   K1
    C   D key1 key2
0  C0  D0   K0   K0
1  C1  D1   K1   K0
2  C2  D2   K1   K0
3  C3  D3   K2   K0
'''
res = pd.merge(left,right,on = ['key1','key2']) # 此处默认为inner 与 concat不同
'''
    A   B key1 key2   C   D
0  A0  B0   K0   K0  C0  D0
1  A2  B2   K1   K0  C1  D1
2  A2  B2   K1   K0  C2  D2
'''
res = pd.merge(left,right,on = ['key1','key2'],how = 'outer')
'''
     A    B key1 key2    C    D
0   A0   B0   K0   K0   C0   D0
1   A1   B1   K0   K1  NaN  NaN
2   A2   B2   K1   K0   C1   D1
3   A2   B2   K1   K0   C2   D2
4   A3   B3   K2   K1  NaN  NaN
5  NaN  NaN   K2   K0   C3   D3
'''
res = pd.merge(left,right,on = ['key1','key2'],how = 'right')
'''
     A    B key1 key2   C   D
0   A0   B0   K0   K0  C0  D0
1   A2   B2   K1   K0  C1  D1
2   A2   B2   K1   K0  C2  D2
3  NaN  NaN   K2   K0  C3  D3
'''

# indicator
df1 = pd.DataFrame({'col1':[0,1],'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
'''
   col1 col_left
0     0        a
1     1        b
   col1  col_right
0     1          2
1     2          2
2     2          2
'''
res = pd.merge(df1,df2,on = 'col1',how = 'outer',indicator = True)
'''
   col1 col_left  col_right      _merge
0     0        a        NaN   left_only
1     1        b        2.0        both
2     2      NaN        2.0  right_only
3     2      NaN        2.0  right_only
'''
res = pd.merge(df1,df2,on = 'col1',how = 'outer',indicator = 'indicator_column')
'''
   col1 col_left  col_right indicator_column
0     0        a        NaN        left_only
1     1        b        2.0             both
2     2      NaN        2.0       right_only
3     2      NaN        2.0       right_only
'''

# merge by index
left = pd.DataFrame({
	'A': ['A0', 'A1', 'A2'],
	'B': ['B0', 'B1', 'B2']
	},index=['K0', 'K1', 'K2'])
right = pd.DataFrame({
	'C': ['C0', 'C2', 'C3'],
	'D': ['D0', 'D2', 'D3']
	},index=['K0', 'K2', 'K3'])
'''
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2
     C   D
K0  C0  D0
K2  C2  D2
K3  C3  D3
'''
res = pd.merge(left,right,left_index = True,right_index = True,how = 'outer')
'''
      A    B    C    D
K0   A0   B0   C0   D0
K1   A1   B1  NaN  NaN
K2   A2   B2   C2   D2
K3  NaN  NaN   C3   D3
'''

# handle overlapping
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
'''
   age   k
0    1  K0
1    2  K1
2    3  K2
   age   k
0    4  K0
1    5  K0
2    6  K3
'''
res = pd.merge(boys,girls,on = 'k',suffixes = ['_boys','_girls'],how = 'inner')
'''
   age_boys   k  age_girls
0         1  K0          4
1         1  K0          5
'''

# join


# pandas可视化
import matplotlib.pyplot as plt
# plot data
# Series
data = pd.Series(np.random.randn(1000),index = np.arange(1000))
data = data.cumsum()
# data.plot()
# plt.show()

# DataFrame
data = pd.DataFrame(np.random.randn(1000,4),index = np.arange(1000),columns = list("ABCD"))
data = data.cumsum()
# data.plot()
# plt.show()

# plot method
# 'bar','hist','box','kde','area','scatter','hexbin','pie'
ax = data.plot.scatter(x = 'A',y = 'B',color = 'DarkBlue',label = 'Class 1')
data.plot.scatter(x = 'A',y = 'C',color = 'DarkGreen',label = 'Class 2',ax = ax)
#plt.show()

