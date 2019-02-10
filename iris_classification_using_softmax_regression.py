import tensorflow as tf
import csv
import numpy as np

#train data
data = np.loadtxt('./iris_training.csv', delimiter = ',', skiprows=1, dtype = np.float32)

x_val = data[:, 0:-1]
y_val = data[:, [-1]]
y_val = y_val.astype(int)
y_val_onehot = np.zeros(shape=[len(y_val), 3])
y_val_onehot[np.arange(len(y_val)), y_val[:,0]] = 1
y_val_onehot = y_val_onehot.astype(int)

#test data
test = np.loadtxt('./iris_test.csv', delimiter = ',', skiprows = 1, dtype = np.float32)
x_test = test[ : , 0:-1]
y_test = test[ : , [-1]]
y_test = y_test.astype(int)
y_test_onehot = np.zeros(shape = [len(y_test), 3])
y_test_onehot[np.arange(len(y_test)), y_test[ :, 0]] = 1
y_val_onehot = y_val_onehot.astype(int)

#placeholders
x = tf.placeholder(tf.float32, shape=[None, 4]) #4 features
y = tf.placeholder(tf.float32, shape=[None, 3]) #3 Species

#1st layer (softmax)
W = tf.Variable(tf.zeros(shape=[4, 3]))
b = tf.Variable(tf.zeros(shape = [3]))
logits = tf.matmul(x, W) + b #logits are the output values before adapting softmax function.
y_pred = tf.nn.softmax(logits)

#loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#model run
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(0, 1000):
    sess.run(train_step, feed_dict = {x: x_val, y: y_val_onehot})
    print(i)

correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy is %.2f%%.'% (sess.run(accuracy, feed_dict = {x: x_test, y: y_test_onehot})*100)) 






#tensorflow way to read csv
'''
filename_queue = tf.train.string_input_producer(
	['./iris_train.csv'], shuffle = False, name = 'filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.]]*5
data = tf.decode_csv(value, record_defaults = record_defaults)

print(data.values)
'''

