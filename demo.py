#coding:UTF-8
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
"""
1、利用pickle实现读写数据

game_data = {"Position":"N2 E3", "Pocket":["Key","Knife"], "Money":160}
save_file = open("save.dat","wb")
pickle.dump(game_data,save_file)
save_file.close()

load_file = open("save.dat","rb")
load_game_data = pickle.load(load_file)
load_file.close
print(load_game_data)
"""

"""
2、利用tensorflow实现计算图搭建和会话输出

x = np.array([[1.0, 2.0]])
w = np.array([[3.0], [4.0]])
y = tf.matmul(x,w)
print(y)
with tf.Session() as sess:
    print(sess.run(y))
"""

"""
#1、定义输出和权重参数
x = tf.placeholder(tf.float32,shape=[None,2]) #如果已知数据个数则None替换为已知数据数，否则利用None进行替代
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed=1))
#2、定义前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
#3、用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("the result is: \n " , sess.run(y,feed_dict={x: [[0.7,0.5],[0.3,0.4],[0.7,0.9]]}))
    print("w1: \n ", sess.run(w1))
    print("w2: \n ", sess.run(w2))
"""

"""
#加入了反向传播进行优化的代码
Batch_Size = 8
Seed = 23455
#基于seed产生随机数
rng = np.random.RandomState(Seed)
#随机数返回32行2列的矩阵，表示32组 体积和重量 作为输入数据集
X = rng.rand(32,2)
#从这个32行2列的矩阵中，取出一行，判断如果和小于1，给y赋值为1，否则为0
#作为输入数据的标签（正确答案）
Y = [[int(X0 + X1 < 1)] for (X0,X1) in X]
print("X: \n ", X)
print("Y: \n ", Y)
#1、定义神经网络的输入，参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None,2))
y_= tf.placeholder(tf.float32, shape=(None,1))

w1 = tf.Variable(tf.random_normal([2,3],stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1, seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#2、定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#不同的优化器
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.trian.AdamOptimizer(0.001).minimize(loss)

#3、生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1: \n ", sess.run(w1))
    print("w2: \n ", sess.run(w2))

    #训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i*Batch_Size) % 32
        end = start + Batch_Size
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d trainning step(s), loss on all data is %g ", (i,total_loss))
    #输出训练后的参数的取值
    print(" \n ")
    print("w1: \n ", sess.run(w1))
    print("w2: \n ", sess.run(w2))
"""

"""
#4.1节的实例：酸奶销量的预测
Batch_Size = 8
Seed = 23455

rdm = np.random.RandomState(Seed)
X = rdm.rand(32,2)
Y_= [[X1+X2+(rdm.rand()/10-0.05)] for (X1,X2) in X]

x = tf.placeholder(tf.float32, shape = (None,2))
y_= tf.placeholder(tf.float32, shape = (None,1))
w = tf.Variable(tf.random_normal([2,1], stddev=1, seed = 1))
y = tf.matmul(x,w)

Cost = 1
Frofit = 9
#loss = tf.reduce_mean(tf.square(y-y_)) #均方误差的损失函数
#loss = tf.reduce_sum(tf.where(tf.greater(y,y_), Cost*(y-y_),Frofit*(y_-y))) #自定义损失函数
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)) #交叉熵的损失定义，Cross Entropy
loss = tf.reduce_mean(ce)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*Batch_Size) % 32
        end = start + Batch_Size
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i %500 == 0:
            total_loss = sess.run(loss,feed_dict={x:X,y_:Y_})
            print("the train step is %d , the total loss is %g ,the w is " % (i,total_loss))
            print(sess.run(w))
"""

"""
#4.2 损失函数的定义方式
1、均方距离的损失函数：
    loss = tf.reduce_mean(tf.square(y-y_))
2、自定义损失函数：
    loss = tf.reduce_sum(tf.where(tf.greater(y,y_), Cost*(y-y_),Frofit*(y_-y)))
3、交叉熵的损失函数-Cross Entropy：
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(ce)
"""

"""
#4.3 学习率的使用 - 指数下降的学习率
LEARNING_RATE_BASE = 0.1 #最初学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减率
LEARNING_RATE_STEP = 1 #喂入多少轮Batch_Size后，更新一次学习率，一般认为：总样本数/Batch_Size

#运行了几轮Batch_size的计数器，初值给0，设为不被训练
global_step = tf.Variable(0,trainable = False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,
LEARNING_RATE_DECAY,staircase=True)
#定义优化参数，初值为10
w = tf.Variable(tf.constant(5,dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learn_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: glabal step is %f , w is %f , learning rate is %f , loss is %f "
        %(i,global_step_val,w_val,learn_rate_val,loss_val))
"""

"""
#4.3 滑动平均的使用 - 求解某个参数以往值的平均值，像影子一样缓慢追随这该参数的变化
#1、定义变量及滑动平均类
#定义了一个32为浮点变量，初值为0.0，这个代码就是在不断更新w1的值，优化w1参数，滑动平均做了w1的一个影子
w1 = tf.Variable(0, dtype = tf.float32)
#定义num_update（NN的迭代轮数），初值为0， 不可被优化
global_step = tf.Variable(0, trainable = False)
#实例化滑动平均类，给删减率0.99，当前轮数为global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
#ema.apply后的括号里是更新列表，每次运行sess.run(ema_op)时，对更新列表中的元素求滑动平均值
#在实际应用中会用tf.trainable_variables()自动将所有参数汇总为列表
#ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

#2、查看不同迭代中变量的取值变化
with tf.Session() as sess:
    #初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #用ema.average(w1)获取w1的滑动平均值，要运行多个节点，作为列表中的元素列出，卸载sess.run()中
    #打印出当前的的参数w1和w1的滑动平均值
    print(sess.run([w1,ema.average(w1)]))

    #参数w1的值赋值为1
    sess.run(tf.assign(w1,1))
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))

    #更新step和w1的值，模拟100轮后，w1为10
    sess.run(tf.assign(w1,10))
    sess.run(tf.assign(global_step,100))
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))

    #每次sess.run 都会更新一次w1的滑动平均值
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))
"""

#4.4 正则化
BATCH_SIZE = 30
seed = 2
#基于随机种子产生随机数
rdm = np.random.RandomState(seed)
X = rdm.randn(300,2)
#print(X)
Y_=[int(X0*X0+X1*X1 < 2) for (X0,X1) in X] #Y_是个一维张量，即一维数组
#print(Y_)
Y_c = [['red' if y else 'blue'] for y in Y_] #Y_c是个二维张量，为1*300

X = np.vstack(X).reshape(-1,2)
Y_= np.vstack(Y_).reshape(-1,1) #这是Y为二维张量，300*1 其中-1代表n行
print(X)
print(Y_)
print(Y_c)
#用plt.scatter画出数据集X中第0列和第1列元素的的点，即（X0，X1），用各行Y_c的颜色（c：colour）
plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
plt.show

#定义神经网络输入，参数，输出，定义前向传播过程
def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape = shape))
    return b

x = tf.placeholder(dtype = tf.float32, shape = (None,2))
y_= tf.placeholder(dtype = tf.float32, shape = (None,1))

w1 = get_weight([2,11],0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x,w1) + b1) #relu非线性函数

w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y = tf.matmul(y1,w2) + b2 #输出层不过激活函数relu

#定义损失函数
loss_mes = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mes + tf.add_n(tf.get_collection('losses'))

#定义反向传播方法，不含正则化项
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mes)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y_[start:end]})
        if i % 1000 ==0:
            loss_mes_val = sess.run(loss_mes,feed_dict = {x:X,y_:Y_})
            print("After %d steps, loss is: %f \n " % (i,loss_mes_val))
    #xx在-3到3之间以步长0.01，yy在-3到3之间以步长0.01，生成二维网格坐标点集合
    xx,yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    #将xx，yy拉直，合并成为一个两列的矩阵，得到一个网格坐标点即集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    #将网格点喂入神经网络，probes为输出
    probes = sess.run(y,feed_dict={x:grid})
    #将probes的shape调整成为xx的样子
    probes = probes.reshape(xx.shape)
    print("w1 is: \n ", sess.run(w1))
    print("b1 is: \n ", sess.run(b1))
    print("w2 is: \n ", sess.run(w2))
    print("b2 is: \n ", sess.run(b2))

    plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
    plt.contour(xx,yy,probes,levels = [.5])
    plt.show()

#定义反向传播算法，使用正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y_[start:end]})
        if i % 1000 ==0:
            loss_mes_val = sess.run(loss_total,feed_dict = {x:X,y_:Y_})
            print("After %d steps, loss is: %f \n " % (i,loss_mes_val))
    #xx在-3到3之间以步长0.01，yy在-3到3之间以步长0.01，生成二维网格坐标点集合
    xx,yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    #将xx，yy拉直，合并成为一个两列的矩阵，得到一个网格坐标点即集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    #将网格点喂入神经网络，probes为输出
    probes = sess.run(y,feed_dict={x:grid})
    #将probes的shape调整成为xx的样子
    probes = probes.reshape(xx.shape)
    print("w1 is: \n ", sess.run(w1))
    print("b1 is: \n ", sess.run(b1))
    print("w2 is: \n ", sess.run(w2))
    print("b2 is: \n ", sess.run(b2))

    plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
    plt.contour(xx,yy,probes,levels = [.5])
    plt.show()
