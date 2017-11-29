import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import read_data


num_feature = 5
hm_epochs = 100
batch_size = 55
rnn_size = 512
output_size = 1
learning_rate = 0.0002

inputs = tf.placeholder('float', [None, 2, num_feature], name = 'inputs')
targets = tf.placeholder('float', name = 'targets')

weight = tf.Variable(tf.truncated_normal([rnn_size, 2]), name = 'weight')
bias = tf.Variable(tf.constant(0.1, shape=[2]),name = 'bias')

training_X, training_Y, testing_X, testing_Y = read_data.aline_data('/Users/oceanland/Downloads/E/Stanford/1.1/CS229/project/own/BIWI_building.csv', num_feature)

#learning_rate = tf.placeholder(tf.float32, [None], name = 'learning_rate')






def recurrent_neural_network(inputs, w, b):
	cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

	outputs, last_State = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32, scope = "dynamic_rnn")
	print(outputs)
	outputs = tf.transpose(outputs, [1, 0, 2])
	print(outputs)
	print(int(outputs.get_shape()[0]))
	last_output = tf.gather(outputs, 1, name="last_output")
	#last_output = tf.gather(outputs, -1, name = "last_output")
	#last_output = outputs[-1]
	print(last_output)
	
	prediction = tf.matmul(last_output, w) + b
	print(prediction)
	return prediction

def train_neural_network(inputs):



    # Extract the x-coordinates and y-coordinates from the target data
    #[x_data, y_data] = tf.split(1, 2, flat_target_data)
    

    prediction = recurrent_neural_network(inputs, weight, bias)
    print(prediction)
    cost = tf.square(tf.norm(prediction - targets))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_epoch_loss = 1.0
        prev_train_loss = 0.0
        iteration = 0
        while (abs(train_epoch_loss - prev_train_loss) > 1e-5):
            iteration += 1
            prev_train_loss = train_epoch_loss
            train_epoch_loss = 0
            for batch in range(int(len(training_X)/batch_size)):
                x_batch, y_batch = read_data.next_batch(batch, batch_size, training_X, training_Y)
                data_feed = {inputs: x_batch, targets: y_batch}

                _, c = sess.run([optimizer, cost], data_feed)
                train_epoch_loss += c
            train_epoch_loss = train_epoch_loss/(int(len(training_X)/batch_size)*batch_size)
            print('Train iteration', iteration,'train loss:',train_epoch_loss)

        trained_w = sess.run(weight)
        trained_b = sess.run(bias)
        print('trained_w: ', trained_w, 'trained_b: ', trained_b, 'trained_w shape: ', trained_w.shape)

        test_epoch_loss = 0
        test_prediction = np.empty([int(len(testing_X)/batch_size)*batch_size, 2])
        for batch in range(int(len(testing_X)/batch_size)):
            x_batch, y_batch = read_data.next_batch(batch, batch_size, testing_X, testing_Y)
            data_feed = {inputs: x_batch, targets: y_batch}

            pre, c = sess.run([prediction, cost], data_feed)
            pre = np.array(pre)
            test_epoch_loss += c
            test_prediction[batch*batch_size : (batch+1)*batch_size, :] = pre
        test_epoch_loss = test_epoch_loss/(int(len(testing_X)/batch_size)*batch_size)
        test_prediction = np.transpose(test_prediction)
        testing_Y_array = np.transpose(np.array(testing_Y)[0 : int(len(testing_X)/batch_size)*batch_size, :])
        test_prediction_and_real = np.vstack((test_prediction, testing_Y_array))
        np.savetxt("test_prediction_and_real.csv", test_prediction_and_real, delimiter = ",")
        print('Test loss:',test_epoch_loss)

        #return trained_w, trained_b

'''
def test_neural_network(inputs):
	trained_w, trained_b = train_neural_network(inputs)
	prediction_test = recurrent_neural_network(inputs, trained_w, trained_b)
	print(prediction_test)
	cost = tf.square(tf.norm(prediction_test - targets))
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for batch in range(int(len(testing_X)/batch_size)):
                x_batch, y_batch = read_data.next_batch(batch, batch_size, testing_X, testing_Y)
                data_feed = {inputs: x_batch, targets: y_batch}

                pre, c = sess.run([prediction_test, cost], data_feed)
                epoch_loss += c

            print('Test epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
'''
train_neural_network(inputs)










