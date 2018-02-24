import numpy as np
# print(np.version.version)


# 创建数组----------------------------------------------------
a = np.array([[1,2,3],[2,3,4]]) #列表转化为矩阵
# print(a)
'''
[[1 2 3]
 [2 3 4]]
'''

# 指定数据dtype
a = np.array([1,2,3],dtype = np.int)
# print(a.dtype)
'''
int64
'''
# a = np.array([1,2,3],dtype=np.int32)
# a = np.array([1,2,3],dtype=np.float)
# a = np.array([1,2,3],dtype=np.float32)

# 创建特定数组
a = np.zeros((3,4)) # 3rows 4cols
a = np.ones((3,4),dtype = np.int)
a = np.empty((3,4)) # 数据为空（很小的数字）
a = np.arange(10,20,2)
'''
[10 12 14 16 18]
'''
a = np.arange(12).reshape(3,4)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
a = np.linspace(1,10,20).reshape(5,4) # 1到10 分割成20个数据
'''
[[  1.           1.47368421   1.94736842   2.42105263]
 [  2.89473684   3.36842105   3.84210526   4.31578947]
 [  4.78947368   5.26315789   5.73684211   6.21052632]
 [  6.68421053   7.15789474   7.63157895   8.10526316]
 [  8.57894737   9.05263158   9.52631579  10.        ]]
'''


# 基础运算----------------------------------------------------
a = np.array([10,20,30,40])
b = np.arange(4)
c = a + b # array([10,21,32,43])
c = a - b
c = b ** 2 # array([0,1,4,9])
c = a * b # 分别相乘
c_dot = np.dot(a,b) # 矩阵相乘 
c_dot = a.dot(b) # 矩阵相乘
c = 10 * np.sin(a) # array([-5.44021111  9.12945251 -9.88031624  7.4511316 ])
b < 3 # array([ True  True  True False])

# 行列操作运算
a = np.random.random((2,4))
# print(a)
'''
[[ 0.94667835  0.23327442  0.69600122  0.83438483]
 [ 0.31039697  0.58137012  0.33945547  0.48471768]]
'''
# print(np.sum(a)) # 4.42627906437
# print(np.min(a)) # 0.23327442341
# print(np.max(a)) # 0.946678351687
# print(np.sum(a,axis = 1)) # [ 2.71033882  1.71594025]
# print(np.min(a,axis = 0)) # [ 0.31039697  0.23327442  0.33945547  0.48471768]
# print(np.max(a,axis = 1)) # [ 0.94667835  0.58137012]

a = np.arange(2,14).reshape(3,4)
# print(np.argmin(a)) #  0
# print(a.argmax(axis = 1)) # [3 3 3]
# print(a.mean()) # 7.5
# print(np.average(a)) # 7.5
# print(np.median(a)) # 7.5
# print(a.cumsum()) # [ 2  5  9 14 20 27 35 44 54 65 77 90]
# print(np.diff(a))
'''
[[1 1 1]
 [1 1 1]
 [1 1 1]]
'''
# print(a.nonzero()) # (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))
a = np.arange(14,2,-1).reshape(3,4)
# print(np.sort(a))
'''
[[11 12 13 14]
 [ 7  8  9 10]
 [ 3  4  5  6]]
'''
# print(np.transpose(a))
# print(a.T)
'''
[[14 10  6]
 [13  9  5]
 [12  8  4]
 [11  7  3]]
'''
# print(np.clip(a,5,9))
'''
[[9 9 9 9]
 [9 9 8 7]
 [6 5 5 5]]
'''


# 索引----------------------------------------------------
a = np.arange(3,15)
a[3] # 6
a = a.reshape(3,4)
'''
[[ 3  4  5  6]
 [ 7  8  9 10]
 [11 12 13 14]]
'''
a[2][1] # 12
a[1,-1::-2] # [10  8]

for i in a.T:
	# print(i)
	'''
	[ 3  7 11]
	[ 4  8 12]
	[ 5  9 13]
	[ 6 10 14]
	'''
	pass

a.flatten() # [ 3  4  5  6  7  8  9 10 11 12 13 14]
#a.flatten() != a.flat 后者用在for中 不可直接打印


# 合并----------------------------------------------------
a = np.array([1,1,1])
b = np.array([2,2,2])
c = np.vstack((a,b)) # vertical stack 可以大于2个合并
'''
[[1 1 1]
 [2 2 2]]
'''
d = np.hstack((a,b)) # horizontal stack  [1 1 1 2 2 2]
e = a[:,np.newaxis] # 变为列向量 不能直接通过a.T来实现
f = np.concatenate((e,b[:,np.newaxis],e),axis = 1)
'''
[[1 2 1]
 [1 2 1]
 [1 2 1]]
'''


# 分割----------------------------------------------------
a = np.arange(12).reshape(3,4)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
b = np.split(a,2,axis = 1) # 不均匀分割造成报错  不均匀分割用np.array_split()
# 或 b = np.hsplit(a,2)
'''
[array([[0, 1],
       [4, 5],
       [8, 9]]), 
 array([[ 2,  3],
       [ 6,  7],
       [10, 11]])]
'''
c = np.split(a,3,axis = 0) # [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
# 或 c = np.vsplit(a,3)

# copy & deep copy----------------------------------------------------
a = np.arange(4)
b = a # 完全复制 内容+地址
c = a
d = b

a[0] = 11
b is a # True
c[0] # 11
d is a # True

e = a.copy() # deepcopy 只复制内容
a[0] = 111
e is a # False
