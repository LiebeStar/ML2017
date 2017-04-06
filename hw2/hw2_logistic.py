import sys
import numpy as np
from math import log, floor

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def load_data():
    X_train = np.delete( np.genfromtxt(sys.argv[1], delimiter=',') , 0, 0)
    Y_train = np.genfromtxt(sys.argv[2], delimiter=',')
    
    X_test = np.delete(np.genfromtxt(sys.argv[3], delimiter=','), 0, 0)
    return X_train, Y_train, X_test

def feature_normalize(X_train, X_test):
    # feature normalization with all X
    X_all = np.concatenate((X_train, X_test))
    mu = np.mean(X_all, axis=0)
    sigma = np.std(X_all, axis=0)
	
    # only apply normalization on continuos attribute
    index = [0, 1, 3, 4, 5]
#    index = []
#    for i in range(0, 106):
#        index.append(i)
    
    mean_vec = np.zeros(X_all.shape[1])
    std_vec = np.ones(X_all.shape[1])
    mean_vec[index] = mu[index]
    std_vec[index] = sigma[index]

    X_all_normed = (X_all - mean_vec) / std_vec

    # split train, test again
    X_train_normed = X_all_normed[0:X_train.shape[0]-1]
    X_test_normed = X_all_normed[X_train.shape[0]:]

    return X_train_normed, X_test_normed

def shuffle(X, Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize], Y[randomize])

def train(X_train_normed, Y_train):
	# parameter initiallize
    w = np.zeros((106,))
    b = np.zeros((1,))
    l_rate = 0.03
    lamda = 0.05
    epoch_num = 10000
    batch_size = 80
    train_data_size = X_train_normed.shape[0]
    batch_num = int(floor(train_data_size / batch_size))
    display_num = 20
    # train with batch
    for epoch in range(epoch_num):
        # random shuffle
        X_train_normed, Y_train = shuffle(X_train_normed, Y_train)
        epoch_loss = 0.0
        for idx in range(batch_num):
            X = X_train_normed[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
    			
            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)
    			
            cross_entropy = -(np.dot(Y, np.log(y)) + np.dot((1.0 - Y), np.log(1.0 - y)))
            epoch_loss += cross_entropy
    			
            w_grad = np.sum(-1 * X * (Y - y).reshape((batch_size,1)), axis=0) + 2.0 * lamda * w
            b_grad = np.sum(-1 * (Y - y))
    
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad
    
        if (epoch+1) % display_num == 0:
            print ('avg_loss in epoch %d : %f' % (epoch+1, (epoch_loss / train_data_size)))

    return w, b

def predict(w, b, X_test_normed):
	# output prediction to 'prediction.csv'
    z = (np.dot(X_test_normed, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    with open(sys.argv[4], 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))
    return

def main():
    X_train, Y_train, X_test = load_data()
    X_train_normed, X_test_normed = feature_normalize(X_train, X_test)
    w, b = train(X_train_normed, Y_train)
    predict(w, b, X_test_normed)
    return

if __name__ == '__main__':
	main()