import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import read_data

num_feature = 5
learning_rate = 0.0005
K = 1000

training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = read_data.aline_data('Crowds_university.csv', num_feature)

def find_KNN(grouped, need): # grouped: total training_X, need: one sample
	dist_list = []
	#print(len(grouped))
	for i in range(len(grouped)):
		x_diff = need[0]-grouped[i][0]
		y_diff = need[1]-grouped[i][1]
		dist = np.sqrt(x_diff.dot(x_diff)+y_diff.dot(y_diff))
		dist_list.append(dist)
	index_list = sorted(range(len(dist_list)), key=lambda k: dist_list[k])
	KNN = []
	for j in range(K):
		KNN.append(index_list[j])
	return KNN

def cost_grad(X, Y, theta): # X: len*5*2, Y: len*2*1, theta: 5*1
	#print(theta.shape)
	grad = np.matmul(X, Y-np.matmul(np.transpose(X, (0,2,1)), theta))
	grad = sum(grad)/grad.shape[0]
	#print(grad.shape)
	diff = Y-np.matmul(np.transpose(X, (0,2,1)), theta)
	cost = sum(np.square(np.linalg.norm(diff, ord=None, axis=1)))/2
	return grad, cost

def linear_regression(train_X, train_Y, target_X, target_Y, test_X, test_Y):
	avg_cost = 0
	for i in range(len(target_X)):
		KNN = find_KNN(train_X, target_X[i])
		current_train_X = []
		current_train_Y = []
		for j in KNN:
			current_train_X.append(train_X[j])
			current_train_Y.append(train_Y[j])
		current_train_X = np.reshape(current_train_X, (len(current_train_X),5,2))
		current_train_Y = np.reshape(current_train_Y, (len(current_train_Y),2,1))
		theta = np.zeros((5,1))
		iteration = 0
		train_epoch_loss = 1.0
		prev_train_loss = 0.0
		while (abs(train_epoch_loss - prev_train_loss) > 1e-5):
			iteration += 1
			prev_train_loss = train_epoch_loss
			train_epoch_loss = 0
			grad, cost = cost_grad(current_train_X, current_train_Y, theta)
			cost = cost/current_train_X.shape[0]
			train_epoch_loss = cost
			theta = theta+learning_rate*grad
		#print(theta.shape)
		print('number i: ', i, 'iteration: ', iteration, 'cost: ', train_epoch_loss)
		avg_cost += train_epoch_loss
	avg_cost = avg_cost/len(target_X)
	print('average cost: ', avg_cost)
	test_X = np.reshape(test_X, (len(test_X),5,2))
	test_Y = np.reshape(test_Y, (len(test_Y), 2,1))
	test_Y = np.reshape(test_Y, (test_Y.shape[0], 2))
	test_Y = np.transpose(test_Y)
	pred_test = np.matmul(np.transpose(test_X, (0,2,1)), theta)
	pred_test = np.reshape(pred_test, (pred_test.shape[0],2))
	pred_test = np.transpose(pred_test)
	#print(pred_test.shape)
	#print(test_Y.shape)
	pred_test_real = np.vstack((test_Y, pred_test))
	np.savetxt("pred_test_KNN.csv", pred_test_real, delimiter = ",")

linear_regression(training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y)
