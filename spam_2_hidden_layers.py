import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix

# 2.a. FETCHING DATA
data = pd.read_csv('spambase.data', sep=',')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# print('X shape', X.shape)
# print('y shape', y.shape)
# =================================================================

# 2. PREPROCESSING DATA

# unique = unique.astype(str)
# print(counts)

# plt.bar(unique, counts, width=0.35)
# Pie chart - Distribution of decision classes
explode = [0, 0.2]
fig, ax = plt.subplots(nrows=2, ncols=2)

unique, counts = np.unique(y, return_counts=True)
ax[0, 0].pie(counts/4601, labels=['Non-spam', 'Spam'], colors=['red', 'green'], autopct='%.1f%%', shadow=True, startangle=90)
ax[0, 0].title.set_text('Whole data distribution')


# 2.b. feature selection
scores = mutual_info_classif(X, y)
relevant_feature_indice = [i for i in range(len(scores)) if scores[i] > 0.05]
X_new = np.array([X[:, i] for i in relevant_feature_indice])
X_new = np.resize(X_new, (X_new.shape[1], X_new.shape[0]))
y = np.resize(y, (X_new.shape[0], 1))
# print(X_new.shape)


# X_new = X
# y = np.resize(y, (X_new.shape[0], 1))

# print(X_new.shape, y.shape)

# 2.c. normalize features
def normalize(X):
    squares = np.square(X)
    norm = np.sqrt(np.sum(squares, axis=1))
    indice_of_zero = np.where(norm == 0)
    for i in indice_of_zero[0]:
        norm[i] = 1
    norm = np.resize(norm, (X.shape[0], 1))
    return X/norm

X_new = normalize(X_new)

# 2.d. split data
def split_data(data, num_train_examples):
# def split_data(X_new, y):
    X = data[:, :-1][:]
    y = data[:, -1][:]
    y = np.resize(y, (len(y), 1))

    num = num_train_examples
    X_train = X[:num, :]
    y_train = y[:num, :]
    
    X_test = X[num:, :]
    y_test = y[num:, :]
    return X_train, X_test, y_train, y_test

# np.random.shuffle(X_new)
num_train_examples = int(X.shape[0] * 0.8)
num_test_examples = X.shape[0] - num_train_examples

data = np.concatenate((X_new, y), axis=1)
np.random.shuffle(data)
X_train, X_test, y_train, y_test = split_data(data, num_train_examples)
# ============================================================

# 3. NEURAL NETWORK - BACKPROPAGATION ALGORITHM
# 3.a. network configuration
input_layer = X_train.shape[1]
hidden_layer1 = 15
hidden_layer2 = 7
output_layer = 1

learning_rate = 0.0001
num_epochs = 20

cost = 0 
costs = []
min_cost = 20000

final_X_train = X_train
final_X_test = X_test
final_y_train = y_train
final_y_test = y_test

# activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#  derivative
def der(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))
# ========

# 3.b. initialize weights and biases
def initialize(shape1, shape2):
    return np.resize(np.random.randn(shape1, shape2), (shape1, shape2))

W1 = initialize(hidden_layer1, input_layer)
B1 = initialize(hidden_layer1, 1)
W2 = initialize(hidden_layer2, hidden_layer1)
B2 = initialize(hidden_layer2, 1)
W3 = initialize(output_layer, hidden_layer2)
B3 = initialize(output_layer, 1)

W1 = normalize(W1)
W2 = normalize(W2)
W3 = normalize(W3)

final_W1 = W1
final_W2 = W2
final_W3 = W3

final_B1 = B1
final_B2 = B2
final_B3 = B3

# =========

# 3.c. train model 
def forward_propagation(X, W1, B1, W2, B2, W3, B3):
    hidden1_net = np.dot(W1, X) + B1
    A1 = sigmoid(hidden1_net)

    hidden2_net = np.dot(W2, A1) + B2
    A2 = sigmoid(hidden2_net)

    output_net = np.dot(W3, A2) + B3
    A3 = sigmoid(output_net)
    y_pred = np.where(A3 >= 0.5, 1, 0)

    return hidden1_net, A1, hidden2_net, A2, output_net, y_pred

def errors(output_net, hidden2_net, hidden1_net, y_desired, y_pred):
    errors_output = np.multiply(der(output_net), y_desired - y_pred)
    errors_hidden2 = np.multiply(der(hidden2_net), np.dot(W3.T, errors_output))
    errors_hidden1 = np.multiply(der(hidden1_net), np.dot(W2.T, errors_hidden2))
    return errors_output, errors_hidden2, errors_hidden1 

# train model
for i in range(num_epochs):
    # data = np.concatenate((X_new, y), axis=1)
    np.random.shuffle(data)
    X_train, X_test, y_train, y_test = split_data(data, num_train_examples)
    # print("X[0]", X_train[0, :])

    for j in range(num_train_examples):
        X_tmp = X_train[j, :]

        y_desired = y_train[j, :]
        X_tmp.shape = (input_layer, 1)
        
        hidden1_net, A1, hidden2_net, A2, output_net, y_pred = forward_propagation(X_tmp, W1, B1, W2, B2, W3, B3)
        errors_output, errors_hidden2, errors_hidden1 = errors(output_net, hidden2_net, hidden1_net, y_desired, y_pred)

        # update
        W1 = W1 + learning_rate * np.dot(errors_hidden1, X_tmp.T)
        B1 = B1 + learning_rate * errors_hidden1
        W1 = normalize(W1)

        W2 = W2 + learning_rate * np.dot(errors_hidden2, A1.T)
        B2 = B2 + learning_rate * errors_hidden2
        W2 = normalize(W2)

        W3 = W3 + learning_rate * np.dot(errors_output, A2.T)
        B3 = B3 + learning_rate * errors_output
        W3 = normalize(W3)



    # calculate cost
    hidden1_net, A1, hidden2_net, A2, output_net, y_pred = forward_propagation(X_train.T, W1, B1, W2, B2, W3, B3)
    y_pred = np.resize(y_pred, (num_train_examples, 1))

    print("EPOCHS ", i)

    cost = 1/2 * np.sum(np.square(y_train - y_pred))
    costs.append(cost)

    if cost < min_cost:
        min_cost = cost
        
        final_W1 = W1
        final_W2 = W2
        final_W3 = W3

        final_B1 = B1
        final_B2 = B2
        final_B3 = B3

        # final_X_train = X_train
        # final_X_test = X_test
        # final_y_train = y_train
        # final_y_test = y_test

    print("COST:", cost)
    print()


print("MIN COST: ", min_cost)
print()

def accuracy(y_pred, y_desired):
    return 1 - np.sum(np.abs(y_desired - y_pred)) / num_train_examples

hidden1_net, A1, hidden2_net, A2, output_net, y_pred = forward_propagation(final_X_train.T, final_W1, final_B1, final_W2, final_B2, final_W3, final_B3)
y_pred = np.resize(y_pred, (num_train_examples, 1))
print("TRAINING SET: \n\taccuracy =", accuracy(y_pred, final_y_train)* 100, "%")

train_confusion = confusion_matrix(final_y_train, y_pred, labels=[0, 1])
print("\tconfusion matrix = ", train_confusion)

nonspam_accuracy = train_confusion[0, 0] / (train_confusion[0, 0] + train_confusion[0, 1])
spam_accuracy = train_confusion[1, 0] / (train_confusion[1, 0] + train_confusion[1, 1])
print("Class non-spam accuracy: ", (nonspam_accuracy * 100), "%")
print("Class spam accuracy: ", (spam_accuracy * 100), "%")

hidden1_net, A1, hidden2_net, A2, output_net, y_test_pred = forward_propagation(final_X_test.T, final_W1, final_B1, final_W2, final_B2, final_W3, final_B3)
print()
y_test_pred = np.resize(y_test_pred, (num_test_examples, 1))
print("TEST SET: \n\taccuracy", accuracy(y_test_pred, final_y_test) * 100, "%")

test_confusion = confusion_matrix(final_y_test, y_test_pred, labels=[0, 1])
print("\tconfusion matrix = ", test_confusion)

nonspam_accuracy = test_confusion[0, 0] / (test_confusion[0, 0] + test_confusion[0, 1])
spam_accuracy = test_confusion[1, 0] / (test_confusion[1, 0] + test_confusion[1, 1])
print("Class non-spam accuracy: ", (nonspam_accuracy * 100), "%")
print("Class spam accuracy: ", (spam_accuracy * 100), "%")

# plot learning curve
ax[0, 1].set_title('Learning curve')
ax[0, 1].set_xlabel('epochs')
ax[0, 1].set_ylabel('cost')
ax[0, 1].grid(color='black', linestyle='-', linewidth = 0.1)

ax[0, 1].plot(np.arange(num_epochs), np.asarray(costs), linestyle='-', linewidth = 0.5)

# plot train and test set distribution
unique, counts = np.unique(y_train, return_counts=True)
ax[1, 0].pie(counts, labels=['Non-spam', 'Spam'], colors=['red', 'green'], autopct='%.1f%%', shadow=True, startangle=90)
ax[1, 0].title.set_text('Train set distribution')

unique, counts = np.unique(y_test, return_counts=True)
ax[1, 1].pie(counts, labels=['Non-spam', 'Spam'], colors=['red', 'green'], autopct='%.1f%%', shadow=True, startangle=90)
ax[1, 1].title.set_text('Test set distribution')

plt.tight_layout()
plt.show()

