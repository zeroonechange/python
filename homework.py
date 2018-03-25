#螺旋矩阵：给定一个m * n要素的矩阵。按照螺旋顺序，返回该矩阵的所有要素
# function output the first row -> end column -> end row -> first column
# output a row or a column and delete it
import numpy as np
output = []
def func_output(array, k):
    if k == 0:
        output.extend(list(array[0,:]))
        array = np.delete(array, 0, 0)
    elif k == 1:
        output.extend(list(array[:,-1]))
        array = np.delete(array, -1, 1)
    elif k == 2:
        output.extend(reversed(list(array[-1,:])))
        array = np.delete(array, -1, 0)
    else:
        output.extend(reversed(list(array[:,0])))
        array = np.delete(array, 0, 1)
    return array

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
k = 0
while arr.size:
    k = k % 4
    arr = func_output(arr, k)
    k = k+1
else :
    print output

#[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
# 需要熟悉这个numpy的api才能往下走


#用栈（使用list）实现队列：支持push(element)，pop()和top()方法。pop和top方法都应该返回第一个元素的值。比如执行以下操作序列：push(1)，pop()，push(2)，push(3)，top()，pop()，你应该返回1，2和2。
class queue:
    def __init__(self):
        self.list = []

    def push(self, element):
        self.list.append(element)

    def pop(self):
        print self.list[0]
        del self.list[0]

    def top(self):
        print self.list[0]


q = queue()
q.push(1)
q.pop()
q.push(2)
q.push(3)
q.top()
q.pop()


#矩阵转换：给定矩阵A，令矩阵B里每个元素B[i][j]的值等于A[0][0]到A[i][j]子矩阵元素的和
import numpy as np
# B = np.array([])
def sum(A):
    B = A.copy()
    for i in range(0,int(A.shape[0])):
        for j in range(0,int(A.shape[1])):
            B[i][j] = np.sum(A[:i+1, :j+1])
    print B
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
sum(arr)
# 还是要熟悉numpy的API


# 翻转单向链表
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def display_list(head):
    vals = []
    while head:
        vals.append(str(head.val))
        head = head.next
    else:
        print '->'.join(vals)


def reverse(head):
    new_head = None               # 新增一个ListNode对象
    while head:
        node = head               # 上一个节点用变量保存
        head = head.next   
        node.next = new_head      # 将保存的节点后面next设置为新的ListNode对象
        new_head = node           # 给新节点赋值
    return new_head


l_5 = ListNode(5)
l_4 = ListNode(4, l_5)
l_3 = ListNode(3, l_4)
l_2 = ListNode(2, l_3)
l_1 = ListNode(1, l_2)
display_list(l_1)
display_list(reverse(l_1))



#利用@property给一个Screen对象加上width和height属性，以及一个只读属性resolution
import traceback


class Screen(object):
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        if isinstance(width, basestring):
            raise ValueError('width can not be string!')
        self._width = width

    @property
    def height(self):
        return self._height

    @width.setter
    def height(self, height):
        if isinstance(height, basestring):
            raise ValueError('height can not be string!')
        self._height = height

    @property
    def resolution(self):
        self._resolution = self._width * self._height
        return self._resolution


s = Screen()
s.width = 75
s.height = 20
print s.resolution
try:
    s.width = 'abc'
except ValueError:
    traceback.print_exc()
try:
    s.resolution = 1500
except AttributeError:
    traceback.print_exc()


#实现一个类，支持用以下方式输出小于1000的所有素数: for i in Prime1000(): print(i)
class Prime(object):
    def __init__(self):
        self.value = 0

    def __iter__(self):
        return self

    def next(self):
        self.value = self.value + 1
        if self.value > 50:
            raise StopIteration()
        # = [self.value for i in range(2, self.value) if (self.value % i) != 0]
        if not [j for j in range(2, self.value) if self.value % j == 0]:
            return self.value

    def __getitem__(self, n):
        if isinstance(n, int):
            if not [j for j in range(2, n) if n % j == 0]:
                return n
            else:
                print '%d in not prime number' % n
        if isinstance(n, slice):
            start = n.start
            stop = n.stop
            L = []
            for x in range(stop):
                if x >= start:
                    if not [j for j in range(2, x) if x % j == 0]:
                        L.append(x)
            return L


# 输出50以内的素数
for value in Prime():     # 这里是个迭代器   会跑 next方法的  
    print value

# 测试支持切片操作  不支持
p = Prime()
p[10:20]


# ---------------------------2017年04月07日00:05:52------ 只是看并未动手----------------------------------#  

# -*- coding: utf-8 -*-
from sklearn import svm, datasets
class Dataset:
    # 下载相关的数据集 并给我们分类好x,y
    def __init__(self, name):
        # 我们有两个选择，一个是'iris'一个是'digits'
        self.name = name
        
    def download_data(self): #下从sklearn载我们指定的数据集 参照官网
        if self.name == 'iris':
            self.downloaded_data = datasets.load_iris()
        elif self.name == 'digits':
            self.downloaded_data = datasets.load_digits()
        else:
            print('Dataset Error: No named datasets')
    
    def generate_xy(self):
        # 通过这个过程来把我们的数据集分为原始数据以及他们的label
        # 我们先把数据下载下来
        self.download_data()
        x = self.downloaded_data.data
        y = self.downloaded_data.target
        print('\n Original data looks like this: \n', x)
        print('\n Labels looks like this: \n', y)
        print('\n the length of row \n ', len(x[0]))   # 64 
        print('\n the length of line \n ', len(x))     # 1797 
        print('\n the length of y \n ', len(y))        # 1797  
        return x,y
    
    def get_train_test_set(self, ratio):
        # 这里，我们把所有的数据分成训练集和测试集
        x, y = self.generate_xy()
        
        n_samples = len(x)          # 1797 
        n_train = int(n_samples * ratio)   # 1257
        # 好了，接下来我们分割数据
        X_train = x[:n_train]
        y_train = y[:n_train]
        X_test = x[n_train:]
        y_test = y[n_train:]
        # 好，我们得到了所有想要的玩意儿
        return X_train, y_train, X_test, y_test
# ====== 我们的dataset类创造完毕=======

# 比如，我们使用digits数据集
data = Dataset('digits')
X_train, y_train, X_test, y_test = data.get_train_test_set(0.7)
clf = svm.SVC()   # 创建一个分类器
clf.fit(X_train, y_train)  # 训练数据
print('the length of X_test', len(X_test))
test_point = X_test[12]
y_true = y_test[12]


print('the oredict value', clf.predict(test_point))
print('the final y_true', y_true)



print('-----go to examine the ration of predict----')    

predict_list = []
for x in X_test:
    predict_list.append(clf.predict(x))
print('predict_result.len', len(predict_list))
print('y_test.len', len(y_test))

# calculate  

if(len(predict_list) != len(y_test)):
    print('something wrong with the calculation, predict_list.len:', len(predict_list), 'y_test.len:', len(y_test))
else:
    cnt = 0
    for i in range(0, len(predict_list) ,1):
        if predict_list[i] == y_test[i]:
            cnt+=1
            

precision_ration = cnt / len(predict_list)
print('precision ration:', precision_ration)  #0.3733222222
with open("somefile.txt", "w") as f:
    f.write( 'precision ration:' + str(precision_ration) )


# -------------------------------------------




