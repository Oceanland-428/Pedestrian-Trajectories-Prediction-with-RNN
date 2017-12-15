'''
This file contains two functions. The first one builds an LSTM RNN model, and the second one used the model to train the parameters and
tests in test set. Before the functions, some variables are defined. These variables can be changed during model evaluation process. This
file can be run directly on terminal line:

python train_test_LSTM.py


Author: Mingchen Li
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
import read_data


num_feature = 5
batch_size = 30
rnn_size = 400
output_size = 1
learning_rate = 0.0005

inputs = tf.placeholder('float', [None, 2, num_feature], name = 'inputs')
targets = tf.placeholder('float', name = 'targets')

weight = tf.Variable(tf.truncated_normal([rnn_size, 2]), name = 'weight')
bias = tf.Variable(tf.constant(0.1, shape=[2]),name = 'bias')

training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = read_data.aline_data('Crowds_university.csv', num_feature)

'''
This function defines a RNN. It is an LSTM RNN for now, but if want to change to GRU, just change the
cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
into
cell = tf.nn.rnn_cell.GRUCell(rnn_size)
'''
def recurrent_neural_network(inputs, w, b):
	cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

	outputs, last_State = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32, scope = "dynamic_rnn")
	outputs = tf.transpose(outputs, [1, 0, 2])
	last_output = tf.gather(outputs, 1, name="last_output")
	
	prediction = tf.matmul(last_output, w) + b
	return prediction

'''
This function trains the model and tests its performance. After each iteration of the training, it prints out the number of iteration
and the loss of that iteration. When the training is done, prints out the trained parameters. After the testing, it prints out the test
loss and saves the predicted values and the ground truth values into a new .csv file so that it is each to compare the results and
evaluate the model performance. The file has two rows, with the first row being predicted values and second row being real values.
'''
def train_neural_network(inputs):
    
    prediction = recurrent_neural_network(inputs, weight, bias)
    #print(prediction.shape)
    #print(tf.reduce_sum(prediction - targets, 0).shape)
    cost = tf.reduce_sum(tf.square(tf.norm(prediction - targets, ord='euclidean', axis=1)))
    #cost = tf.square(tf.norm(tf.reduce_sum(prediction - targets, 0)))		# prediction: (len,2)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_epoch_loss = 1.0
        prev_train_loss = 0.0
        iteration = 0
        train_cost_list = []
        dev_cost_list = []
        while (abs(train_epoch_loss - prev_train_loss) > 1e-5):
            iteration += 1
            prev_train_loss = train_epoch_loss
            #train_epoch_loss = 0
            
            for batch in range(int(len(training_X)/batch_size)):			# There will be some data that's been thrown away if the size of
            																# training_X is not divisible by batch_size
                x_batch, y_batch = read_data.next_batch(batch, batch_size, training_X, training_Y)
                data_feed = {inputs: x_batch, targets: y_batch}
                _, c = sess.run([optimizer, cost], data_feed)
                #print('train: ', c)
                #train_epoch_loss += c/batch_size
            '''
            # training cost
            data_feed = {inputs: training_X, targets: training_Y}
            _, train_c = sess.run([optimizer, cost], data_feed)
            train_epoch_loss = train_c/len(training_X)
            '''
            #train_epoch_loss = train_epoch_loss/int(len(training_X)/batch_size)		# Use the same expression as above to make
            																		# sure not count the data that is thrown away
            


            dev_epoch_loss = 0
            for batch in range(int(len(dev_X)/batch_size)):
                x_batch, y_batch = read_data.next_batch(batch, batch_size, dev_X, dev_Y)
                data_feed = {inputs: x_batch, targets: y_batch}
                c = sess.run(cost, data_feed)
                #print('dev: ', c)
                dev_epoch_loss += c/batch_size
            dev_epoch_loss = dev_epoch_loss/int(len(dev_X)/batch_size)
            # training cost
            train_epoch_loss = 0
            for batch in range(int(len(training_X)/batch_size)):
                x_batch, y_batch = read_data.next_batch(batch, batch_size, training_X, training_Y)
                data_feed = {inputs: x_batch, targets: y_batch}
                c = sess.run(cost, data_feed)
                #print('dev: ', c)
                train_epoch_loss += c/batch_size
            train_epoch_loss = train_epoch_loss/int(len(training_X)/batch_size)

            # dev cost
            
            '''
            data_feed = {inputs: dev_X, targets: dev_Y}
            _, dev_c = sess.run([prediction, cost], data_feed)
            dev_epoch_loss = dev_c/len(dev_X)
            '''
            train_cost_list.append(train_epoch_loss)
            dev_cost_list.append(dev_epoch_loss)
            print('Train iteration', iteration,'train loss:',train_epoch_loss)
            print('Train iteration', iteration,'dev loss:',dev_epoch_loss)
        iter_list = range(1, iteration+1)
        plt.figure(1)
        plt.plot(iter_list, train_cost_list)
        plt.plot(iter_list, dev_cost_list)
        plt.title('iteration vs. epoch cost, university')
        plt.show()

        # After the training, print out the trained parameters
        trained_w = sess.run(weight)
        trained_b = sess.run(bias)
        #print('trained_w: ', trained_w, 'trained_b: ', trained_b, 'trained_w shape: ', trained_w.shape)

        # Begin testing
        test_epoch_loss = 0
        test_prediction = np.empty([int(len(testing_X)), 2])
        '''
        data_feed = {inputs: testing_X, targets: testing_Y}
        pre, test_c = sess.run([prediction, cost], data_feed)
        test_prediction = pre
        test_epoch_loss = test_c/int(len(testing_X))
        '''
        test_prediction = np.empty([int(len(testing_X)/batch_size)*batch_size, 2])
        for batch in range(int(len(testing_X)/batch_size)):
            x_batch, y_batch = read_data.next_batch(batch, batch_size, testing_X, testing_Y)
            data_feed = {inputs: x_batch, targets: y_batch}
            pre, c = sess.run([prediction, cost], data_feed)
            pre = np.array(pre)
            test_epoch_loss += c
            test_prediction[batch*batch_size : (batch+1)*batch_size, :] = pre
        test_epoch_loss = test_epoch_loss/(int(len(testing_X)/batch_size)*batch_size)
        
        print('Test loss:',test_epoch_loss)

        # Save predicted data and ground truth data into a .csv file.
        test_prediction = np.transpose(test_prediction)																# The first row of file: prediction
        testing_Y_array = np.transpose(np.array(testing_Y)[0 : int(len(testing_X)/batch_size)*batch_size, :])		# The second row of file: ground truth
        test_prediction_and_real = np.vstack((test_prediction, testing_Y_array))
        np.savetxt("LSTM_test_prediction_and_real.csv", test_prediction_and_real, delimiter = ",")


train_neural_network(inputs)


