import numpy as np


num_feature = 5


def preprocess(file_path):
    '''
    The function that pre-processes the pixel_pos.csv files of each dataset
    into data that can be used
    params:
    data_dirs : List of directories where raw data resides
    data_file : The file into which all the pre-processed data needs to be stored
    '''
    # all_ped_data would be a dictionary with mapping from each ped to their
    # trajectories given by matrix 3 x numPoints with each column
    # in the order x, y, frameId
    # Pedestrians from all datasets are combined
    # Dataset pedestrian indices are stored in dataset_indices
    all_ped_data = {}
    current_ped = 0

    # Define the path to its respective csv file
    #file_path = '/Users/oceanland/Downloads/E/Stanford/1.1/CS229/project/Dataset/BIWI\ Walking\ Pedestrians\ dataset\ at\ ETH\ main\ building.csv'

    # Load data from the csv file
    # Data is a 4 x numTrajPoints matrix
    # where each column is a (frameId, pedId, y, x) vector
    data = np.genfromtxt(file_path, delimiter=',')

    # Get the number of pedestrians in the current dataset
    numPeds = np.size(np.unique(data[1, :]))

    # For each pedestrian in the dataset
    for ped in range(1, numPeds+1):
        # Extract trajectory of the current ped
        traj = data[:, data[1, :] == ped]
        # Format it as (x, y, frameId)
        traj = traj[[3, 2], :]

        # Store this in the dictionary
        all_ped_data[current_ped + ped] = traj
    return all_ped_data

def aline_data(file_path, num_feature):
    all_ped_data = preprocess(file_path)
    for pedID, _ in all_ped_data.copy().items():
        if all_ped_data[pedID].shape[1] <= num_feature:
            del all_ped_data[pedID]
    
    same_size_data_X = []
    same_size_data_Y = []
    '''
    for pedID in all_ped_data:
        hm_group = all_ped_data[pedID].shape[1]/(num_feature+1)
        for i in range(hm_group):
            data_X_0 = all_ped_data[pedID][0, i*(num_feature+1) : (i+1)*(num_feature+1)-1]
            for _ in range(len(data_X_0)):
                data_X_0[_] = np.array(data_X_0[_])
                print(data_X_0[_].shape)
            data_X_1 = all_ped_data[pedID][1, i*(num_feature+1) : (i+1)*(num_feature+1)-1]
            for _ in range(len(data_X_1)):
                data_X_1[_] = np.array([data_X_1[_]])
            data_Y_0 = all_ped_data[pedID][0, i*(num_feature+1)+num_feature]
            data_Y_0 = np.array([data_Y_0])
            data_Y_1 = all_ped_data[pedID][1, i*(num_feature+1)+num_feature]
            data_Y_1 = np.array([data_Y_1])

            same_size_data_X.append(data_X_0)
            same_size_data_X.append(data_X_1)

            same_size_data_Y.append(data_Y_0)
            same_size_data_Y.append(data_Y_1)
    '''
    for pedID in all_ped_data:
        hm_group = all_ped_data[pedID].shape[1]/(num_feature+1)
        for i in range(hm_group):

            same_size_data_X.append(all_ped_data[pedID][:, i*(num_feature+1) : (i+1)*(num_feature+1)-1])

            same_size_data_Y.append(all_ped_data[pedID][:, i*(num_feature+1)+num_feature])

        training_X = []
        training_Y = []
        testing_X = []
        testing_Y = []

        for j in range(len(same_size_data_X)):
        	if j < len(same_size_data_X)*0.7:
        		training_X.append(same_size_data_X[j])
        		training_Y.append(same_size_data_Y[j])
    		else:
    			testing_X.append(same_size_data_X[j])
    			testing_Y.append(same_size_data_Y[j])

    return training_X, training_Y, testing_X, testing_Y


def next_batch(batch, batch_size, filt_X, filt_Y):
    '''
    Function to get the next batch of points
    '''
    # List of source and target data for the current batch
    x_batch = []
    y_batch = []
    # For each sequence in the batch
    for i in range(batch_size):

        x_batch.append(filt_X[batch*batch_size+i])
        y_batch.append(filt_Y[batch*batch_size+i])

    return x_batch, y_batch



training_X, training_Y, testing_X, testing_Y = aline_data('/Users/oceanland/Downloads/E/Stanford/1.1/CS229/project/own/BIWI_building.csv', num_feature)
print(len(training_Y))
print(len(testing_Y))