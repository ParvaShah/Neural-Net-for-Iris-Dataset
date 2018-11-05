#####################################################################################################################
#   
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   
#
#####################################################################################################################


import numpy as np
import pandas as pd


class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        #raw_input = pd.read_csv(train)
        raw_input = train
        
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
      
        nrows = len(train_dataset.index)
        
        
        self.X = train_dataset.iloc[:, 0:(ncols -3)].values.reshape(nrows, ncols-3)
        self.y = train_dataset.iloc[:, (ncols-3):].values.reshape(nrows, 3)
        #
        # Find number of input and output layers from the dataset
        #
        
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])
            
        

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "relu":
            self.__relu(self, x)
        if activation == "tanh":
            self.__tanh(self, x)

    #
    
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "relu":
            self.__relu_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)

    def __sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))
    def __relu(self, x):
        return np.fmax(0,x)
    def __tanh(self, x):
        return ((np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    def __relu_derivative(self, x):
        
        if x.all==0:
            return 0
        else:
            return 1
       
        
    def __tanh_derivative(self, x):
        return  (1 - x*x)

    #
    

    def preprocess(self, X):
        column_names_for_onehot = X.columns[:]
        # One hot encoding using pd.get_dummies
        X1 = pd.get_dummies(X, columns=[column_names_for_onehot[-1]], drop_first=False)
        #standardization of data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X1)
        scaler.transform(X1)
        
        return X1

    # Below is the training function

    def train(self, max_iterations = 1000000, learning_rate = 0.0001):
        for iteration in range(max_iterations):
            activate='tanh'
            out = self.forward_pass(activate) # added an parameter so that various types of activation can be used
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation='tanh')
          
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)
            
           
                

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        #print(self.w01)
        #print(self.w12)
        #print(self.w23)

    def forward_pass(self,activate='sigmoid'):
        # pass our inputs through our neural network
        if activate=='sigmoid':
            
            in1 = np.dot(self.X, self.w01 )
            
            self.X12 = self.__sigmoid(in1)
            
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        
        if activate=='relu':
            
            in1 = np.dot(self.X, self.w01 )
            
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        
        if activate=='tanh':
            
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        self.deltaOut = delta_output

    

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

   

    def predict(self, test, header = True):
        
        #raw_input = pd.read_csv(test)
        raw_input = test
        
        test_dataset = self.preprocess(raw_input)
       
        ncols = len(test_dataset.columns)
        
        nrows = len(test_dataset.index)
        
        self.X = test_dataset.iloc[:, 0:(ncols -3)].values.reshape(nrows, ncols-3)
        self.y = test_dataset.iloc[:, (ncols-3):].values.reshape(nrows, 3)
        
        out = self.forward_pass(activate='tanh')
        
        error = 0.5 * np.power((out - self.y), 2)
        
        
        
      
        
        
        
        return np.sum(error)


if __name__ == "__main__":
    
   
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
    
    from sklearn.utils import shuffle
    df = shuffle(df)
    x=df[:90]
    y=df[90:]
    neural_network = NeuralNet(x)
    neural_network.train()
    print('trained')
    testError = neural_network.predict(y)
    print('test-error' , str((testError)))

