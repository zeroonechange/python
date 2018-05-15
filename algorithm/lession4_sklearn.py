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
print('precision ration:', precision_ration)  #0.3733
with open("somefile.txt", "w") as f:
	f.write( 'precision ration:' + str(precision_ration) )



