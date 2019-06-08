import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

#feature selections
data = pd.read_csv('spambase.data', sep=',').values
X = data[:, :-1]
y = data[:, -1]
# print(X.shape)

scores = mutual_info_classif(X, y)
relevant_feature_indice = [i for i in range(len(scores)) if scores[i] > 0.05]
X_new = np.array([X[:, i] for i in relevant_feature_indice])
X_new = np.resize(X_new, (X_new.shape[1], X_new.shape[0]))
y = np.resize(y, (X_new.shape[0], 1))

# X_new = X
# y = np.resize(y, (X_new.shape[0], 1))

# print(X_new.shape, y.shape)

# normalize features
def normalize(X):
    squares = np.square(X)
    norm = np.sqrt(np.sum(squares, axis=1))
    indice_of_zero = np.where(norm == 0)
    for i in indice_of_zero[0]:
        norm[i] = 0.0001
    norm = np.resize(norm, (X.shape[0], 1))
    return X/norm

X_new = normalize(X_new)

data = np.concatenate((X_new, y), axis=1)
np.random.shuffle(data)

# split data
def split_data(data):
# def split_data(X_new, y):
    X = data[:, :-1][:]
    y = data[:, -1][:]
    y = np.resize(y, (len(y), 1))

    num = int(X.shape[0] * 0.8)
    X_train = X[:num, :]
    y_train = y[:num, :]
    
    X_test = X[num:, :]
    y_test = y[num:, :]
    return X_train, X_test, y_train, y_test

# np.random.shuffle(X_new)
X_train, X_test, y_train, y_test = split_data(data)

# activation function
# def tangensoid(z):
#     return 2 / (1 + np.exp(-z)) - 1

# def der(z):
#     return 1 / 2 * (1 - np.square(tangensoid(z)))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#  derivative
def der(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

# network configuration
input_layer = X_train.shape[1]
hidden_layer1 = 15
hidden_layer2 = 7
hidden_layer3 = 3
output_layer = 1

num_train_examples = X_train.shape[0]

# initial weights and biases
def initialize(shape1, shape2):
    return np.resize(np.random.randn(shape1, shape2), (shape1, shape2))

W1 = initialize(hidden_layer1, input_layer)
B1 = initialize(hidden_layer1, 1)
W2 = initialize(hidden_layer2, hidden_layer1)
B2 = initialize(hidden_layer2, 1)
W3 = initialize(hidden_layer3, hidden_layer2)
B3 = initialize(hidden_layer3, 1)

W4 = initialize(output_layer, hidden_layer3)
B4 = initialize(output_layer, 1)

def forward_propagation(X, W1, B1, W2, B2, W3, B3, W4, B4):
    hidden1_net = np.dot(W1, X) + B1
    A1 = sigmoid(hidden1_net)

    hidden2_net = np.dot(W2, A1) + B2
    A2 = sigmoid(hidden2_net)

    hidden3_net = np.dot(W3, A2) + B3
    A3 = sigmoid(hidden3_net)

    output_net = np.dot(W4, A3) + B4
    A4 = sigmoid(output_net)

    A4 = np.where(A4 >= 0.5, 1, 0)

    return hidden1_net, A1, hidden2_net, A2, hidden3_net, A3, output_net, A4

def errors(output_net, hidden2_net, hidden1_net, hidden3_net, y_desired, y_pred):
    errors_output = np.multiply(der(output_net), y_desired - y_pred)
    errors_hidden3 = np.multiply(der(hidden3_net), np.dot(W4.T, errors_output))
    errors_hidden2 = np.multiply(der(hidden2_net), np.dot(W3.T, errors_hidden3))
    errors_hidden1 = np.multiply(der(hidden1_net), np.dot(W2.T, errors_hidden2))
    return errors_output, errors_hidden3, errors_hidden2, errors_hidden1 

# train model
W1_init = W1
cost = 0 
costs = []

learning_rate = 0.001
num_epochs = 20
min = 2000

final_W1 = W1
final_W2 = W2
final_W3 = W3
final_W4 = W4

final_B1 = B1
final_B2 = B2
final_B3 = B3
final_B4 = B4

final_X_train = X_train
final_X_test = X_test
final_y_train = y_train
final_y_test = y_test

for i in range(num_epochs):
    np.random.shuffle(data)
    X_train, X_test, y_train, y_test = split_data(data)
    # print("X[0]", X_train[0, :])

    for j in range(num_train_examples):
        X_tmp = X_train[j, :]

        y_desired = y_train[j, :]
        X_tmp.shape = (input_layer, 1)
        
        hidden1_net, A1, hidden2_net, A2, hidden3_net, A3, output_net, y_pred = forward_propagation(X_tmp, W1, B1, W2, B2, W3, B3, W4, B4)
        errors_output, errors_hidden3, errors_hidden2, errors_hidden1  = errors(output_net, hidden2_net, hidden1_net, hidden3_net, y_desired, y_pred)
        
        # update
        W1 = W1 + learning_rate * np.dot(errors_hidden1, X_tmp.T)
        B1 = B1 + learning_rate * errors_hidden1
        W1 = normalize(W1)

        W2 = W2 + learning_rate * np.dot(errors_hidden2, A1.T)
        B2 = B2 + learning_rate * errors_hidden2
        W2 = normalize(W2)

        W3 = W3 + learning_rate * np.dot(errors_hidden3, A2.T)
        B3 = B3 + learning_rate * errors_hidden3
        W3 = normalize(W3)

        W4 = W4 + learning_rate * np.dot(errors_output, A3.T)
        B4 = B4 + learning_rate * errors_output
        W4 = normalize(W4)



    # calculate cost
    hidden1_net, A1, hidden2_net, A2, hidden3_net, A3, output_net, y_pred = forward_propagation(X_train.T, W1, B1, W2, B2, W3, B3, W4, B4)
    y_pred = np.resize(y_pred, (num_train_examples, 1))

    print("EPOCHS ", i)

    cost = 1/2 * np.sum(np.square(y_train - y_pred))
    costs.append(cost)

    if cost < min:
        min = cost
        final_W1 = W1
        final_W2 = W2
        final_W3 = W3
        final_W4 = W4

        final_B1 = B1
        final_B2 = B2
        final_B3 = B3
        final_B4 = B4

        final_X_train = X_train
        final_X_test = X_test
        final_y_train = y_train
        final_y_test = y_test

    print("COST:", cost)
    print()

# print result 

print("MIN COST: ", min)

def accuracy(y_pred, y_desired):
    return 1 - np.sum(np.abs(y_desired - y_pred)) / num_train_examples

hidden1_net, A1, hidden2_net, A2, hidden3_net, A3, output_net, y_pred = forward_propagation(final_X_train.T, final_W1, final_B1, final_W2, final_B2, final_W3, final_B3, final_W4, final_B4)
y_pred = np.resize(y_pred, (num_train_examples, 1))
print("TRAIN SET: ", accuracy(y_pred, final_y_train)* 100, "%")

hidden1_net, A1, hidden2_net, A2, hidden3_net, A3, output_net, y_test_pred = forward_propagation(final_X_test.T, final_W1, final_B1, final_W2, final_B2, final_W3, final_B3, final_W4, final_B4)
y_test_pred = np.resize(y_test_pred, (len(y_test_pred), 1))
print("TEST SET: ", accuracy(y_test_pred, final_y_test) * 100, "%")

plt.title('Learning curve')
plt.xlabel('epochs')
plt.ylabel('cost')
plt.grid(color='black', linestyle='-', linewidth = 0.1)

plt.plot(np.arange(num_epochs), np.asarray(costs), linestyle='-', linewidth = 0.5)
plt.show()


