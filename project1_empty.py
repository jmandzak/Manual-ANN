import numpy as np
import sys
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.num_inputs = input_num
        self.lr = lr

        # if weights is passed, use it. If not, just use random list of weights with len = num_inputs
        if type(weights) != type(None):
            self.weights = weights
        else:
            self.weights = [1 for x in range(input_num)]
        
        # bias is passed as last weight
        # set bias then remove it from weights
        self.bias = weights[-1]
        self.weights = self.weights[:-1]
        
    #This method returns the activation of the net
    # linear = 0
    # log = 1
    def activate(self,net):
        # if linear, just return
        if self.activation == 0:
            return net
        else:
            return 1/(1+np.exp(-net))
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        sum = 0
        for i, w in zip(input, self.weights):
            sum += i * w

        # add in bias and do activation function
        sum += self.bias
        result = self.activate(sum)

        self.output = result
        return result

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        print('activationderivative')   
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        print('calcpartialderivative') 
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        print('updateweight')

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.num_neurons = numOfNeurons
        self.activation = activation
        self.num_inputs = input_num
        self.lr = lr

        if type(weights) != type(None):
            self.weights = weights
        else:
            self.weights = np.empty((self.num_neurons, self.num_inputs))

        # initialize all the neurons and add them to a list of neurons for the class to keep
        self.all_neurons = []
        for i in range(self.num_neurons):
            n = Neuron(activation, input_num, lr, self.weights[i])
            self.all_neurons.append(n)
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        vec = []
        for neuron in self.all_neurons:
            vec.append(neuron.calculate(input))

        return vec
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas') 
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.num_layers = numOfLayers
        self.num_neurons = numOfNeurons # this is a vector of neurons per layer
        self.num_inputs = inputSize
        self.activation = activation # again, a vector of activations for each layer
        self.loss = loss
        self.lr = lr

        
        if type(weights) != type(None):
            self.weights = weights

        # TODO - figure out how to make 3d numpy array of different dimensions
        # else:
            # weights = []
            # for i in range(self.num_layers):
            #     middle = []
            #     for j in range(self.num_neurons[i]):
            #         w = []
            #         if i == 0:
            #             for k in range(self.num_inputs):
            #                 w.append(0.5)
            #         else:
            #             for k in range(self.num_neurons[i-1]):
            #                 w.append(0.5)
            #         middle.append(np.array(w))
            #     weights.append(np.array(middle))
            # self.weights = np.array(weights)

        # else:
        #     self.weights = np.empty((self.num_inputs))
        #     for i in range(self.num_layers):
        #         if i == 0:
        #             self.weights[i] = np.empty((self.num_neurons[i], self.num_inputs))
        #         else:
        #             self.weights[i] = np.empty((self.num_neurons[i], self.num_neurons[i-1]))

        # create list of layers to hold on to
        self.all_layers = []
        for i in range(self.num_layers):
            if i == 0:
                layer = FullyConnected(self.num_neurons[i], self.activation[i], self.num_inputs, self.lr, self.weights[i])
            else:
                layer = FullyConnected(self.num_neurons[i], self.activation[i], self.num_neurons[i-1], self.lr, self.weights[i])

            self.all_layers.append(layer)
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        print(input)
        for i in range(self.num_layers):
            input = self.all_layers[i].calculate(input)
            print(input)
    
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    # making the assumption that both yp and y are vectors considering MSE we assume multiple output neurons
    # and binary cross entropy we need multiple values to calculate the actual loss
    def calculateloss(self,yp,y):

        # MSE loss
        if self.loss == 0:
            sum = 0
            for i in range(len(yp)):
                val = y[i] - yp[i]
                val = val ** 2
                sum += val

            sum = sum / len(yp)
            return sum

        # otherwise binary cross entropy
        else:
            sum = 0
            for i in range(len(yp)):
                val = -1 * (y[i] * np.log(yp[i]) + (1 - y[i]) * (np.log(1-yp[i])))
                sum += val

            sum = sum / len(yp)
            return sum
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    # this simply does it for one neuron
    def lossderiv(self,yp,y):
        
        # MSE loss
        if self.loss == 0:
            # out - target
            return yp - y
        else:
            # negative out / target + (1 - target) / (1 - out)
            val = (-1 * y) / yp
            val = val + ((1 - y) / (1 - yp))
            return val
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        f = NeuralNetwork(2, np.array(([2, 2, 2])), 2, [1, 1, 1], 1, 0.1, np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]]))
        f.calculate([0.05, 0.1])
        #print(f.calculateloss([0.751365069, 0.77292846532], [0.01, 0.99]))
        print(f.calculateloss([.2, .4, .9, .3, .8, .9], [0, 0, 0, 1, 1, 1]))

        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')