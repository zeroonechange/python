# -*- coding: utf-8 -*-
# coding: utf-8

# 单行注释

'''
第一行
第二行
'''

"""
多行注释
在python中 ' 和 " 都是一样的
"""

# 连接行
str = 'abcd' \
	   'efgh'
print(str)

str = 'Hello \n World'
print(str) 

str = """Hello
World"""
print(str)                      # 俩行  


print('abc"123"efg')  	# abc"123"efg
print("abc'123'efg")	               # abc'123'efg
print('abc\'123\'efg')	# abc'123'efg

#----------------------------------#

print(type([1, 2, 3, 'a', 'b']))          # list
print(type((1, 'abc')))	    # tuple
print(type(set(['a', 'b', 3])))	    # set
print(type({'a':1, 'b':2}))               # dict

def func(a, b, c):
	print(a, b, c)
print(type(func))   # function
a = func
print(type(a))		# function

import string
print(string)   # module

# 类和类的实例
class MyClass(object):
	pass
print(type(MyClass))  	# type
my_class = MyClass()
print(my_class)			# __main__.MyClass object at ox101caca10

#  for 循环
for i in range(0, 30, 5):
	print(i)


while 循环
a = 0
i = 1
while i<=100:
	a +=i
	i +=1
print(a)	


#  函数元组
def func_name(arg_1, arg_2):
	print(arg_1, arg_2)		# (1, 2)
	return arg_1, arg_2	
r = func_name(1, 2)
print(type(r))     			# tuple
print(r[0], r[1])


def func(x, y=500):
	print('x=', x)
	print('y=', y)
	return x+y
print(func(100))
print(func( y = 300, x = 200 ))
print(func( x = 400))

def func(p):
	print('x=', p['x'])
	print('y=', p['y'])

print(func({'x':100, 'y':200}))

# 可变参数   *numbers是元组 会将后面的参数打包成一个只读数组
def func(name, *numbers):
	print(type(numbers))
	print(numbers)

func('Tom', 1, 2, 3, 4, 'adb12', 'd')


def func(*args):
	print(args)
	print(type(args))  # tuple

func(1, 2, 3, 'a', 'b', 'c')
func('x=', 100, '; y=', 200)  

# 字典  **kvs means key/values
def func(name, **kvs) :
	print(name)
	print(type(kvs))		# dict
	print(kvs)
func('Tom', china = 'Beijing', uk = 'London')

# 可变参数必须放在后面 且 * 为必须带名字
def func(a, b, c, *, china, uk):
	print(china, uk)
func(1, 2, 3, china = 'BJ', uk = 'LD')    # 2.7.X 貌似不支持


def func(a, b, c=0, *args, **kvs):
	print(a, b, c)
	print(args)
	print(kvs)
func(1, 2)
func(1, 2, 3)
func(1, 2, 3, 'a', 'b', 'c')
func(1, 2, 3, 'a', 'b', china = 'BJ', uk = 'LD')
func(1, 2, 3, *('a', 'b'), **{'china' : 'BJ', 'uk' : 'LD'})


# 递归问题
def my_sum(i):
	if i < 0 : 
		raise ValueError
	elif i <= 1 :
		return i
	else :
		return i + my_sum(i-1)

print(my_sum(1))
print(my_sum(5))
print(my_sum(500))

# f(n) = f(n-1) + f(n-2)

def fib(n):
	if n < 1 :
		raise ValueError
	elif n <=2 :
		return 1
	else :
		return fib(n-1) + fib(n-2)

print(fib(40))

#  函数是可以作为参数
def sum(x, y, p = None ) :
	s = x + y
	if p:
	    p(s)
	return s

sum(100, 200) 
sum(100, 200, print)      # can not run in python2.7x

def cmp(x, y, cp = None) :
 	if not cp :
 		if x > y :
 			return 1
 		elif x < y :
 			return -1
 		else :
 		    return 0
 	else :
 		return cp(x, y)

def my_cp(x, y):
	if x < y :
		return 1
	elif x == y :
		 return 0
	else : 
		return -1
print(cmp(100, 200))
print(cmp(100, 200, my_cp))

print([1,2,3,4], sum)

def do_sum(data , method):
	return method(data)

print(sum([10, 20]))
print(do_sum([1, 2, 3, 4], sum) )


# homework : change '  Hello, how are u? ' to  ' u? are how ,Hello  '
# 思路：先将单词逆序，再将整个句子逆序
def reverse(str_list, start, end):
	while(start < end):
		str_list[start], str_list[end] = str_list[end], str_list[start]
		start += 1
		end -= 1
setence = '  Hello, how are u? '
str_list = list(setence)
i = 0 
while i < len(str_list):
	if str_list[i] != ' ':
		start = i
		end = start + 1
		while (end < len(str_list)) and str_list[end] != ' ':
			end +=1
		reverse(str_list, start, end - 1)
		i = end
	else:
		i += 1
str_list.reverse()
print(''.join(str_list))


# ------------------list-------------------- #
li = [1,2,3, '456', [1,2,3], {1:'one', 2:'two'}]
print(type(list))       
print(type(li))         # list

print(li[0])
print(li[-1])
print(li[-2])

# find index of element
print(li.index('456'))
print(li.index([1,2,3]))
print(li.index(-1)
del(li[-1])  #delete element del(list[index]) 

# add element
l_a = [1, 2, 3]
l_a.append(4)
l_a.append(5)
l_b = [6, 7, 8] 
l_a.extend(l_b)     # extend会展开再一个一个的添加
l_a.append(l_b)   #  append 会将其视为一个对象
print(l_a)              # will be [1, 2, 3, 4, 5, 6, 7, 8]

l_a = []
if not l_a:
	print('Empty')  # empty 有内存分配 
	pass
if l_a is None:
	print('None')   # None 无内存分配
	pass

# for loop
for i in li:
	print(i)
	pass
for i in range(len(li)):
	print(i)
	pass

# ------------------dict-------------------- #
d = {'a':1, 'b':2, 1:'one', 2:'two', 3:[1,2,3]}
print(type(dict))
print(type(d))
print(d)

# # access element
print(d['a'])
print(d[1])
print(d[3])

# key is exist
print('two' in d)
print(3 in d)
del(d[3])   # del(dict[key])
d[3] = [1, 2, 3, 4]       # add or update element when key=3
d['a'] = '1234'     	  # add or update element when key='a'

print(len(d))

# ------------------set-------------------- #
s_a = set([1, 2, 2, 3, 4, 5, 6])
s_b = set([4, 5, 6, 7, 8, 9])
print(s_a)
print(s_b)

# element is exist
print(5 in s_a)
print(10 in s_b)

# 并集   A | B  =>  A.unio(B)
print(s_a | s_b)
print(s_a.union(s_b))

# 交集	A & B  =>  A.intersection(B)
print(s_a & s_b)
print(s_a.intersection(s_b))

# 差集	A - B = A - (A & B)   =>   A.difference(B)
print(s_a - s_b)
print(s_a.difference(s_b))

# 对称差  (A | B ) - (A & B)     =>   A.symmetric_difference(B)
print(s_a ^ s_b)
print(s_a.symmetric_difference(s_b))

# add or modify element
s_a.add('x')
s_a.update([4, 5, 60, 70])   # add array to set
print(s_a)

s_a.remove(70)
print(s_a)
print(len(s_a))

for x in s_a:
	print(x)
	pass

# ------------------slice-------------------- #
# slice 切片 从数组切出另一个数组
li = list(range(10))
print(li)   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# [start, end, step]  || (start - end) 要和 step的正负号一致

print(li[2:5])	# [2, 3, 4]
print(li[:4])	# [0, 1, 2, 3]
print(li[5:])	# [5, 6, 7, 8, 9]
print(li[0:20:3])        # [0, 3, 6, 9]


# how about minus
print(li[5:-2])            # [5, 6, 7]
print(li[9:0:-1])	# [9, 8, 7, 6, 5, 4, 3, 2, 1]
print(li[9:0:1])	# []
print(li[9::-1])	# [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(li[::-2])	# [9, 7, 5, 3, 1]

# a new object
print(li)
re_li = li[::-1]            # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(re_li)


# ------------------comprehension-------------------- #
# comprehension  推导列表
# simple case
li = []
for i in range(10):
	li.append(i) 
print(li)

li = list(range(10)):
        print(li)

li  = [1] * 10
print(li)

# 浅拷贝
li_2d = [[0] * 3 ] * 3
print(li_2d)

li_2d[0][0] = 100
print(li_2d)

#深拷贝
li_2d = [ [0] * 3 for i in range(3)]
print(li_2d)

li_2d[0][0] = 100
print(li_2d)

li = (x for x in range(10))
print(type(li))         # generator
print(li)                  # generator object
      
for i in range(10):  # way1 
      print(next(li))
      
for i in li:                # way2
      print(i)

      
li = [x for x in range(10)]    
print(type(li))         # list
print(li)                  # [1, 2, 3, 4]

li =  {x for x in range(3)}
print(type(li))         # set
print(li)                  # {0, 1, 2}

s = {x for x in range(10)  if x%2==0 }
print(type(s))     # set
print(s)              # {0, 8, 2, 4, 6}

s = [ x%2==0 for x in range(10)]
print(type(s))     # list
print(s)              # [True, False, Ture, False,Ture, False, Ture, False, Ture, False]

d = {x: x % 2 == 0 for x in range(10)}
print(type(d))    # list
print(d)             # {0: True, 1: False, 2: True, 3: False, 4: True, 5: False, 6: True, 7: False, 8: True, 9: False}

# so  'x for x in range(10)'  is a comprehension


# generator 将真正的计算推迟到使用时  不一次性生成很多元素，省内存
# 2.7 版本时一次性生成100W个数字，在3.5版本并不是真正生成100W个数字而是在next取值时才生成
print(type(range(10)))   # type  

# 平方表
square_table = []
for i in range(50000):
	square_table.append(i * i)
for i in range(5):
	print(square_table[i])

square_generator = ( x * x for x in range(50000))
print(type(square_generator))   # generator

for i in range(5):
	print(next(square_generator))

def fib(limit):
	n, a, b = 0 , 0 , 1
	while n < limit:
		yield b
		a, b  = b, a + b
		n += 1
	pass
import traceback

f = fib(5)
print(type(f))
print(next(f))
print(next(f))
print(next(f))
print(next(f))
print(next(f))
try: 
	print(next(f))
except StopIteration:
	traceback.print_exc()
for i in fib(5):
	print(i)


# Iterable  Iterator
# 可迭代 和 迭代器不一样的概念， 可迭代表示可以用for循环， 而迭代器是用来使用next()不断返回下一个值，采用惰性计算
# 生成器一定是迭代器  使用一个生成一个 看下面的fib例子
from collections import Iterable
from collections import Iterator

print(isinstance([1,2,3], Iterable))	                # True
print(isinstance({}, Iterable))			# True
print(isinstance(123, Iterable))		# False
print(isinstance('abc', Iterable))		# True

print(isinstance([1, 2, 3], Iterator))                         # False
 
g = (x * x for x in range(10))
print(type(g))		# <type 'generator'> 
print(isinstance(g, Iterable))	# True
print(isinstance(g, Iterator))	# True
for i in g:
	print(i)

def fib(limit):
	n, a, b = 0 , 0 , 1
	while n < limit:
		yield b
		a, b  = b, a + b
		n += 1
	pass
f = fib(5)
print(type(f))
print(isinstance(f, Iterable))   # True
print(isinstance(f, Iterator))	# True
for i in f:
	print(i)



# 面向对象  
# 成员属性名称前 加上 __ 意为private
# get / set ：  get_name()   set_name(name) 
class Student:
	def __init__(self, name, age):
		self.name = name
		self.age = age
		
	def detail(self):
		print(self.name)
		print(self.age)
      
class PrimaryStudent(Student):  # inherent
	def lol(self):
		print('can not win then run faster than others')

class CollegeStudent(Student):
	def __init__(self, name, age, gf):  # overrite构造函数
		self.name = name
		self.age = age
		self.gf = gf

	def gf_detail(self):
		print(self.gf)

obj1 = PrimaryStudent('小学生', 7)
obj1.lol()
obj1.detail()

obj2 = CollegeStudent('王思聪', 29, '张雨欣')
obj2.detail()
obj2.gf_detail()

print(dir(obj1))                            # class info as list
print(hasattr(obj1, 'name'))        # True
setattr(obj1, 'name', 'jack')  
print(getattr(obj1, 'name'))          # jack
print(getattr(obj1, 'name', 404))  # jack
fn = getattr(obj1, 'detail')             #7
fn()

#  实例属性和类属性
class Student(object):
	name = 'Student'
	def __init__(self, name):
		self.name = name   # 类属性
s = Student('Bob')
s.score = 90			   # 实例属性

print(s.name) 
s.name = 'Jack'          # 给实例属性绑定name属性, 实例属性优先级比类属性高 
print(s.name)              # Jack
print(Student.name)   # Student
del s.name                 # 删除实例name属性
print(s.name)             # Student


import sys
print(sys.path)

# ---------------文件----------------------------
# 第一种直接方式
file1 = open("test.txt")
file2 = open("output.txt", "w")  # w 表示 write (覆写) r 表示 read  a 表示 append (追写)
while True:
	line = file1.readline()
	file2.write('"' + line[:s] + '"' + ",")
	if not line:
		break
file1.close()
file2.close()

# read()  将文本文件所有行读到一个字符串中
# readline() 一行一行的读
# readlines() 将文本所有行读到一个list中，每一行是list的一个元素

# 第二种  文件迭代器
file2 = open("output.txt", "w")
for line in open("test.txt"):
	file2.write('"' + line[:s] + '"' + ",")

# 第三种  文件上下文管理器
# 打开文件
with open("somefile.txt", "r") as f:
	data = f.read()

# loop 整个文档
with open("somefile.txt", "w") as f:
	for line in f:
		#  处理每一行

# 写入文本
with open("somefile.txt", "w") as f:
	f.write("xxx")
	f.write("xxx")

# 要把打印的line写入文件中
with open("somefile.txt", "w") as f :
	print(line1, file=f)
	print(line2, file=f)

# 二进制文件读写
f = open("EDC.jpg", "rb")
print(f.read())  # 输出\xff\xd8.... 十六进制表示的字节

# 任何非标准文本文件(py2标准是ASCII， py3是unicode)，用二进制读入文件，用.decode() 来解码
f = open("DeGuangGuo.txt", "rb")
u = f.read().decode('DeyunCode')

# 文件和目录的操作
# python调用内置的os模块来调用操作系统的接口函数
import os
os.name  # posix == nix   nt == windows
os.uname()  # 查看具体信息

# 环境变量 存在os.environ中  是list

# 当前目录的绝对路径
os.path.abspath('.')
# 在某个目录下创建一个新目录，把新目录表示出来
os.path.join('/Users/EDC', 'Pictures') # 得到是新路径的字符串
# 创建目录
os.mkdir('/Users/EDC/Pictures/')
# 删除目录
os.rmdir('/Users/EDC/Pictures')
# 拆分字符串
os.path.split('/Users/EDC/Pictures/AJ.avi') # 拆分为俩部分， 后一部分为最后级别的目录或者文件
# ('/Users/EDC/Pictures/', 'AJ.avi')
# 得到文件扩展名
os.path.splitext('/Users/EDC/Pictures/AJ.avi')
# ('/Users/EDC/Pictures/AJ', '.avi')
# 文件重命名
os.rename('xxx.xx', 'bbb')
# 删除文件
os.remove('xxx')

# 可以使用 Shutil来帮助我们搞定文件
# 列出当前目录下的所有目录
[x for x in os.listdir('.') if os.path.isDir(x)]
# 列出 .py文件
[x for x in os.listdir('.') if os.path.isDir(x) and os.path.splitext(x)[1] == '.py']


# 序列化 从内存存储到硬盘或者传输的过程为序列化  从硬盘到内存为反序列
import pickle
d = dict(name='jack', age=23, score=60)
str = pickle.dumps(d) # 调用pickle的dumps函数进行序列化处理
print(str)

f = open("dump.txt", "wb")
pickle.dump(d, f)	# 将内容序列化写到文件中
f.close()


# 反序列化
import pickle
f = open("dump.txt", "rb")
d = pickle.load(f)    # 调用load做反序列化
f.close()
print(d)
print('name is %s' % d['name'])

# python2 和3 里面的pickle不一致，为了保证和谐
try:
	import cPickle as pickle
except ImportError:
	import pickle

# json 序列化   使用json这个库即可 
import json
d1 = dict(name='jack', age = 29, score=32)
str = json.dump(d1)  # 序列化

d2 = json.loads(str) # 反序列化



# 高阶函数  可以把别的函数作为参数传入的函数叫高阶函数
def add(x, y, f):
	return f(x) + f(y)
add(-5, 6, abs)         # 11

# 匿名函数  python使用lambda来创建匿名函数  
sum = lambda arg1, arg2 : arg1 + arg2
sum(10, 20) # 30 

# reduce 内建函数是个二元操作函数， 用来将一个数据集合所有数据进行二元操作
# 先对集合第1，2 个数据进行func()操作，得到的结果与第三个数据用func()运行，如此最后得到一个结果
# 顾名思义就是reduce将一个list缩成一个值
from functools import reduce
l = [1,2,3,4,5]
print(reduce(lambda x, y: x-y , 1))
# x 开始的时候赋值为10， 然后依次
print(reduce(lambda x, y: x-y, l, 10))

# map 应用于每一个可迭代的项返回一个结果list，map函数会把每一个参数都以相应的处理函数进行迭代处理
# 本质就是将原有的list根据lambda法则变成另一个list
l = [1, 2, 3]
new_list = list(map(lambda i: i+1, l))
# 变成了 [2, 3, 4]

l2 = [4, 5, 6]
new_list = list(map(lambda x, y : x + y, l, l2)) 
# 变成了 [5, 7, 9]

# filter 对序列进行过滤处理 
l = [100, 20, 24, 50, 110]
new = list(filter(lambda x : x < 50 , l))
# [20, 24]

# 装饰器  和测试方法中的@before @test @end 类似  可以带参 和 多个装饰器 
# 简单来说，你处理一个方法时需要统一做某件事
from functools import wraps

def makeHtmlTag(tag, *args, **kwds):
      def real_decorator(fn):             # fn is hello()
            css_class = " class='{0}'".format(kwds["css_class"]) \
			if "css_class" in kwds else ""
            def wrapped(*args, **kwds):
                  return "<" + tag + css_class + ">" + fn(*args, **kwds) + "</"+tag+">"
            return wrapped
      return real_decorator

@makeHtmlTag(tag="b", css_class="bold_css")
@makeHtmlTag(tag="i", css_class="italic_css")
def hello():
    return "hello world"
 
print(hello())
# <b class='bold_css'><i class='italic_css'>Hello World</i></h>
# 这里包了俩层 b为最外面的那层，i为中间层

# 高效率的递归  这里有个问题就是输入60会超过整数范围从而报错
from functools import wraps
from datetime import  datetime
def memo(fn):
      cache= {}
      miss= object()

      @wraps(fn)
      def wrapper(*args):
            result = cache.get(args, miss)
            if result is miss:
                  result = fn(*args)
                  cache[args] = result
            return result
      return wrapper

@memo
def fib(n):
      if n<2:
            return n
      return fib(n-1) + fib(n-2)

start = datetime.now()
print(fib(40))          # 102334155
end = datetime.now()
print((end - start).microseconds)   # 71061

def fib2(n):
      if n<2:
            return n
      return fib2(n-1) + fib2(n-2)
start1 = datetime.now()
print(fib2(40))         # 102334155
end1 = datetime.now()
print((end1 - start1).microseconds)  # 641741

      


# 偏函数 只设置一部分参数
int('123455')			# 默认转化为10进制
int('123456', base 8)  	                # 8进制
int('123456', base 16)                            # 16进制

# 为了方便不需要每次都输入多少进制的值
def int2(x, base=2):
	return int(x, base)

# 借助functools.partial来创建一个偏函数
import functools
int2 = functools.partial(int, base=2)
print(int2('1000000'))

# 传字典可以默认参数
kw = {'base': 2}
print(int('10010', **kw))

# 传list
args = (10, 5, 6, 7)
print(max(*args))


#------------------高级面向对象-----------------------------------#
# __slots__ 的使用 
import traceback

from types import MethodType

class MyClass(object):
    __slots__ = ['name', 'set_name']

def set_name(self, name):
    self.name = name

cls = MyClass()
cls.name = 'Tom'
cls.set_name = MethodType(set_name, cls)   #动态添加方法
cls.set_name('Jerry')
print(cls.name)
try:
    cls.age = 30
except AttributeError:
    traceback.print_exc()

class ExtMyClass(MyClass):
    pass

ext_cls = ExtMyClass()
ext_cls.age = 30
print(ext_cls.age)


# property的get/set使用, 只读
import traceback
class Student:
    @property
    def score(self):     # 创建了一个score对象 
        return self._score
    
    @score.setter
    def score(self, value):   # score对象的setter方法
        if not isinstance(value, int):
            raise ValueError('not int')
        elif (value < 0 ) or (value > 100):
            raise ValueError('not between 0 ~ 100')
            
        self._score = value
    @property
    def double_score(self):    # 只读属性  
        return self._score * 2

s = Student()
s.score = 75    # 注意看调用的方法属性名称  是s.score 不是 s._score
print(s.score)
try:
    s.score = 'abc'
except ValueError:
    traceback.print_exc()

try:
    s.score = 123
except ValueError:
    traceback.print_exc()

print(s.double_score)

try:
    s.double_score = 150
except AttributeError:
    traceback.print_exc()

# 描述器
'''
实现了__set__ __get__ __del__方法的类称为描述器
python是门动态语言，类的生成都是在编译的时候
'''
class MyProperty:
    def __init__(self, fget=None, fset=None, fdel=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        
    def __get__(self, instance, cls):
        print('__get__')
        if self.fget:
            return self.fget(instance)
            
    def __set__(self, instance, value):
        print('__set__', value)
        if self.fset:
            return self.fset(instance, value)
            
    def __del__(self, instance):
        print('__del__')
        if self.fdel:
            return self.fdel(instance)
            
    def getter(self, fn):
        print('getter')
        self.fget = fn
        
    def setter(self,fn):
        print('setter')
        self.fset = fn
        
    def delete(self, fn):
        self.fdel = fn

class Student:
    @MyProperty   # 生成MyProperty类
    def score(self):
        return self._score
    @score.setter  # 所以会调用MyProperty.setter()
    def set_score(self, value):
        self._score = value
        
s = Student()
s.score = 21   # 调用的是__set__ 方法
print(s.score) # 调用的是__get__ 方法

# 控制类的魔术(内部)函数
'''
 类似于java中的Object的
 toString  ->  __str__
 __iter__ __next__ -> 迭代器内部函数
 支持下标访问
 __name__ 就是函数名称
 call 重载()函数
'''
class Fib100:
    def __init__(self):
        self._1 , self._2 = 0, 1
    def __iter__(self):
        return self
    def __next__(self):
        self._1, self._2 = self._2, self._1 + self._2
        if self._1 > 100:
            raise StopIteration()
        return self._1
    
for i in Fib100():
    print(i)


class Fib:
    def __getitem__(self, n):
        a, b = 1, 1
        for i in range(n):
            a, b = b, a+b
        return a
    
f = Fib()
print(f[1])
print(f[5])
print(f[10])


class Myclass:
    
    def __call__(self):
        print('u can call cls() directly')
        
cls = Myclass()
cls()  # 可调用的

print(callable(cls))   
print(callable(max))
print(callable([1, 2, 3]))
print(callable(None))
print(callable('str'))


# 枚举
from enum import Enum

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr'))
for name, member in Month.__members__.items():
    print(name, '=>', member, ',', member.value)

jan = Month.Jan
print(jan)

# 添加额外的方法和属性
def add(self, value):
    self.append(value)

class ListMetaclass(type):  # 元类一定是从type继承下来的
    def __new__(cls, name, bases, attrs):
        # print(cls)
        # print(name)
        # print(bases)  基类
        # print(type(attrs))
        # attrs['add'] = lambda self, value: self.append(value)
        attrs['add'] = add    # 添加额外的方法add
        attrs['name'] = 'Tom' # 添加额外的属性
        # attrs   就是一张hash表 可以额外添加属性和方法
        return type.__new__(cls, name, bases, attrs)
        
class MyList(list, metaclass = ListMetaclass):  # 额外增加add方法，实际等价于append。
    pass

mli = MyList()
mli.add(1)
mli.add(2)
mli.add(3)
print(mli.name)
print(mli)


# orm框架
class Field:
    def __init__(self, name, col_type):
        self.name = name
        self.col_type = col_type
    
class IntegerField(Field):
    def __init__(self, name):
        super(IntegerField, self).__init__(name, 'integer')

class StringField(Field):
    def __init__(self, name):
        super(StringField, self).__init__(name, 'varchar(1024)')

class ModelMetaClass(type):  # 元类
        def __new__(cls, name, bases, attrs):
            if name=='Model':
                return type.__new__(cls, name, bases, attrs)
            print('Model name : %s' %name)
            mappings = {}       # 新建一个字典来保存键值对 放入attrs哈希表中
            for k, v in attrs.items():  # attrs.items()应该是自定义属性
                if isinstance(v, Field):
                    print('Field name: %s'%k)
                    mappings[k] = v
            for k in mappings.keys():
                attrs.pop(k)  # 不需要了 扔掉
            attrs['__mappings__'] = mappings
            attrs['__table__'] = name
            return type.__new__(cls, name, bases, attrs)
        
class Model(dict, metaclass = ModelMetaClass):
    def __init__(self, **kvs):
        super(Model, self).__init__(**kvs)
        
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'Model' object has no attribute '%s'." % key)
            
    def __setattr__(self, key, value):
        self[key] = value
    
    def save(self):
        fields = []
        params = []
        args = []
        for k, v  in self.__mappings__.items():
            fields.append(v.name)
            params.append('?')
            args.append(getattr(self, k, None))
        sql = 'insert into %s(%s) values(%s)' % (self.__table__, ','.join(fields), ''.join(params))
        print('sql:', sql)
        print('args:', args)        
        
    
class User(Model):
    id = IntegerField('id')
    name = StringField('name')
                    
    
u = User()
u.id = 10
u.name='tom'
u.save()

# 异常
import traceback

try:
    # r = 10 / 0
except ZeroDivisionError as e:
    print(e)
    r = 1
else:
    print('没有异常')
finally:
    print('不管有没有异常都执行')
print(r)

# 单元测试和logging自行百度


from threading import Thread
import time
def my_counter():
    i = 0
    for i in range(100000000):
        i = i + 1
    return True

def main1():
    thread_array = {}
    start_time = time.time()
    for tid in range(2):
        t = Thread(target = my_counter)
        t.start()
        t.join()
    end_time = time.time()
    print('1 Total Time: {}'.format(end_time- start_time))  # 单线程65.8902

def main2():
    thread_array = {}
    start_time = time.time()
    for tid in range(2):
        t = Thread(target = my_counter)
        t.start()
        thread_array[tid] = t
    for i in range(2):
        thread_array[i].join()
    end_time = time.time()
    print('2 Total Time: {}'.format(end_time- start_time))  # 并发线程34.899

if __name__ == '__main__':
    main1()
    main2()


'''
fork操作，调用一次，俩次返回。 因为OS自动把当前进程(父进程)复制了一份(子进程),然后分别在父进程和子进程内返回。
子进程永远返回0，父进程返回子进程ID，子进程通过getppid获取父进程ID
注意: 在windows上不能运行
'''
import os
print('Process (%s) start...' % os.getpid())
pid= os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child prcess (%s).' %(os.getpid(), pid))


'''
multiprocessing是跨平台的多进程模块，提供了一个Process类代表一个进程
'''
from multiprocessing import Process
import time

def f(n):
    time.sleep(1)
    print(n*n)
if __name__ == '__main__':
    for i in range(10):
        p = Process(target = f, args=[i,])
        p.start()    

'''
进程间通信Queue
通过共享变量Queue来通信
'''
from multiprocessing import Process, Queue
import time

def write(q):
    for i in ['A', 'B', 'C', 'D', 'E']:
        print('put %s to queue' % i)
        q.put(i)
        time.sleep(0.5)

def read(q):
    while True:
        v = q.get(True)
        print('get %s from queue' %v)
        
if __name__ == '__main__':
    q = Queue()
    pw = Process(target = write, args=(q,))
    pr = Process(target = read, args=(q,))
    pw.start()
    pr.start()
    pr.join()
    pr.terminal()

'''
进程池 Pool 用于批量创建子进程 灵活控制子进程的数量
'''
from multiprocessing import Pool
import time

def f(x):
    print(x*x)
    time.sleep(2)
    return x*x

if __name__ == '__main__':
    pool = Pool(processes=5)
    res_list=[]
    for x in range(10):
        # 以异步并行的方式启动进程  同步等待使用Pool.apply
        res = pool.apply_async(f, [i,])
        print('------:', i)
        res_list.append(res)
    pool.close()
    pool.join()
    
    for r in res_list:
        print('result', (r.get(timeout=5)))


'''
多个进程之间的内存资源独立的，多线程可以共享一个进程的内存资源
'''
from multiprocessing import Process
import threading
import time
lock = threading.Lock()

def run(info_list, n):
    lock.acquire()
    info_list.append(n)
    lock.release()
    print('%s\n' % info_list)
    
if __name__ == '__main__':
    info=[]
    for i in range(10):
        p = Process(target = run, args=[info, i])
        p.start()
        p.join()
    time.sleep(1)

    print('-------------threading---------------')    
    
    for i in range(10):
        p = threading.Thread(target=run, args=[info, i])
        p.start()
        p.join()


'''
函数式编程
immutable data : 不可变数据
first class function: 函数像变量一样使用
尾递归优化: 每次递归都重用stack
优点: 
    parallelization 并行
    lazy evaluation 惰性求值  
    determinism确定性  

Python中的lambda, map, filter, reduce 
map(function, collection) : 依次执行function(collection[i]) -> List
filter(function, collection): 依次执行function(collection[i]) -> collection
reduce(function, collection, start_value): 依次迭代调用如有start_value则作为初始值
'''
g = lambda x: x * 2
print(g(3))

print((lambda x : x * 2)(6))

name_len = list(map(len, ["I", "am", "you"]))
print(name_len)


items = [1, 2, 3, 4, 5]
squard = list(map(lambda x: x**2, items))
print(squard)

number_list = range(-5, 5)
print(list(number_list))
less_than_zero = list(filter(lambda x: x<0 , number_list))
print(less_than_zero)

import functools

def add(x,y):
    return x+y
print(functools.reduce(add, range(1, 5)))
print(functools.reduce(add, range(1, 5), 10))


# 描述是在干什么而不是怎么干
num = [2, -5, 9, 7, -2, 5, 3, 1, 0, -3, 8]
positive_num = list(filter(lambda x: x>0, num))
#average = functools.reduce(add, positive_num) /len(positive_num)
average = functools.reduce(lambda x,y : x + y , positive_num) /len(positive_num)
print(average)


'''
正则表达式
http://www.cnblogs.com/chuxiuhong/p/5885073.html
'''
import re
m = re.match(r'dog', 'dog cat dog')
print(m.group())

print(re.match(r'cat', 'dog cat dog'))

s = re.search(r'cat', 'dog cat dog')
print(s.group)

print(re.findall(r'dog', 'dog cat doge'))

contactInfo = 'Doe, John: 5555-1212'
m = re.search(r'(\w+), (\w+): (\S+)', contactInfo)
print(m.group(0))
print(m.group(1))
print(m.group(2))
print(m.group(3))

strs = 'purple alice-b@google.com monkey dishwasher'
match = re.search(r'[\w.-]+@[\w.-]+',strs)
if match:
    print(match.group())


'''
enumerate函数
'''
l = [1, 2 ,3]
for index, text in enumerate(l):
    print(index, text)
    
'''
cllections 是python内建的一个集合模块
deque是为了高效的实现插入和删除操作的双向列表，适合队列和栈
OrderedDict会按照插入的顺序排列
Counter是个简单的计数器，是dict的一个子类，有统计了各个元素的个数等API
'''



