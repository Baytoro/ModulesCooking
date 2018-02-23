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
# print(a)
'''
[10 12 14 16 18]
'''
a = np.arange(12).reshape(3,4)
# print(a)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
a = np.linspace(1,10,20).reshape(5,4) # 1到10 分割成20个数据
# print(a)
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
c_dot = np.dot(a,b) #矩阵相乘 
c_dot = a.dot(b) #


