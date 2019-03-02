# Simple Neural Network that learns a double well function. Graphviz is used to visualise the trajectory in weight space.
# Extend by a fractal-like function


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('agg') # different backend so we don't need tkinter
import matplotlib.pyplot as plt
import networkx as nx
#import pygraphviz as pg
import string



def dblWell(x, bias=0): # double well
    E0=1
    E1=0.05
    center_x = 5
    
    x = x-center_x
    return E0*(E1*x**4-x**2+bias*x**3)

def fractalLike(x, L=10): # fractal-like Fourier series from Ann. Stat. 34, 1636
    coeffs = [.21, 1.25, .61, .25, .13, .10, 1.16, .18, .12, .23, .21, .19, .37, .99, .36, .02, .06, .08, .09, .04]
    series = 0.0
    for i,coeff in enumerate(coeffs):
        series += coeff*np.sin(i*2.0*np.pi*x/L)
    return 2.0*series


class singleLayerNet(nn.Module): # from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/301_regression.py
    def __init__(self, n_feature, n_hidden, n_output):
        super(singleLayerNet, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)   # single hidden layer with n_hidden output features
        self.predict = nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
    
class threeLayerNet(nn.Module): # from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/301_regression.py
    def __init__(self, n_feature, n_hidden, n_output):
        super(threeLayerNet, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)   # input layer with n_hidden output features
        self.hidden2 = nn.Linear(n_hidden, n_hidden-20)  # single hidden layer with n_hidden output features
        n_hidden -= 20 # narrow down the NN
        self.hidden3 = nn.Linear(n_hidden, n_hidden)    # single hidden layer with n_hidden output features
        self.predict = nn.Linear(n_hidden, n_output)    # output layer

    def forward(self, x):
        x = F.sigmoid(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)             # linear output
        return x    

def getCurrentWeights(net, indices): # extract weights of layers in indices in form of a python list from a pytorch net
    weightsReturn = []
    for index in indices:
        layerWeights = list(net.parameters())[index][0] # extract weights from parameter set
        layerWeights = layerWeights.data.numpy()        # convert from pytorch tensor to numpy array
        #reluWeights = layerWeights.tolist()            # convert from numpy array to python list
        weightsReturn.append(np.copy(layerWeights))     # return a copy, since usually net.parameters() gives an iterator reference    
    return weightsReturn

def getDistanceMatrix(weights, magnify=1):
    distance = np.zeros((np.shape(weights)[0], np.shape(weights)[0]))
    for t1 in range(np.shape(weights)[0]):
        for t2 in range(t1):
            # sum over element-wise squared differences
            for layer in range(np.shape(weights)[1]):
                distance[t1, t2] += sum(np.square(weights[t1][layer]-weights[t2][layer]))*magnify
                distance[t2, t1] = distance[t1, t2] # this distance metric is symmetric
    return distance

if __name__ == "__main__":
    # sample from function
    N       = 150         # how many total sampled points
    N_train = int(0.8*N) # how many training points
    
    L = 10 # sample points between 0 and this value
    X = L*np.random.rand(N)
    
    def samplingFunction(X):
        return dblWell(X, bias=-5e-2)
        #return fractalLike(X, L)
    y = samplingFunction(X) # choose which function to sample from
    
    X_train = X[:N_train].reshape(-1,1) # training data
    y_train = y[:N_train].reshape(-1,1)
    X_test = X[N_train:].reshape(-1,1)  # validation data
    y_test = y[N_train:].reshape(-1,1)

    
    mode = "torch"
    progress_plots = False
    
    if (mode == "tf"):
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
        model = Sequential()
        # reshape into 2D before input?
        model.add(Dense(3, input_shape=(1,)))
        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
        
        train_model = model.fit(X_train, y_train,
                      batch_size=128,
                      epochs=50,
                      verbose=1,
                      validation_data=(X_test, y_test))
        
    elif (mode == "torch"):
        n_features = 1
        x = torch.from_numpy(X_train).float()
        y = torch.from_numpy(y_train).float()
        
        # characterise neural net
        H1,H2,H3,H4,H5 = 80,60,40,40,20
        layerIndices = [0,2,4,6,8] # used later in weight extraction
        
        net = nn.Sequential(nn.Linear(x.shape[1],H1),    # define the network
                            nn.ReLU(), 
                            nn.Linear(H1, H2), 
                            nn.ReLU(), 
                            nn.Linear(H2, H3), 
                            nn.ReLU(), 
                            nn.Linear(H3, H4), 
                            nn.ReLU(), 
                            nn.Linear(H4, H5), 
                            nn.ReLU(), 
                            nn.Linear(H5, x.shape[1]))
                            #nn.LogSoftmax(dim=1))    
        epochs = 500

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = nn.MSELoss()  # mean squared loss for regression
        
        weights = []
        
        for t in range(epochs):
            prediction = net(x)     # input x and predict yhat based on x
            loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            if t % int(epochs/20) == 0:
                # show learning process
                print('training loss = %.4f' % loss.data.numpy())
                if (progress_plots):
                    plt.cla()
                    plt.scatter(x.data.numpy(), y.data.numpy())
                    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
                    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color':  'red'})
                    plt.savefig('tmp'+str(t)+'.jpg')
                
                # save all current weights for visualisation    
                currentWeights = getCurrentWeights(net, layerIndices)
                weights.append(currentWeights)
                    
        # print loss on test set
        xtest = torch.from_numpy(X_test).float()
        ytest = torch.from_numpy(y_test).float()
        testloss = loss_func(net(xtest), ytest)
        print("-------- test loss = %.4f" % testloss.data.numpy())

        # calculate distance matrix between the sets of weights for every time step
        distance = getDistanceMatrix(weights, magnify=15)
        
        # draw distance graph
        dt = [('len', float)]
        distance = distance.view(dt)
        graph = nx.from_numpy_matrix(distance)
        graph = nx.drawing.nx_agraph.to_agraph(graph)
        graph.node_attr.update(shape="circle", style="filled")
        colourmap = matplotlib.cm.get_cmap('jet')
        for nodeId in graph.nodes():
            node = graph.get_node(nodeId)
            colour = matplotlib.colors.to_hex(colourmap(int(nodeId)/graph.number_of_nodes()))
            node.attr['fillcolor'] = colour
        graph.edge_attr.update(style="invis")  # hide edges
        graph.draw('graph.png', format='png', prog='neato')
        
        # draw prediction
        plotrange = np.linspace(0, L, num=120)
        plt.figure()
        plt.plot(plotrange, samplingFunction(plotrange), label='true fct.')
        plt.plot(X_train, y_train, '.', label='training data')
        plt.plot(X_test, y_test, 'r.', label='test data')
        plt.ylim(-10,15)
        
        prediction = net(Variable(torch.Tensor(plotrange.reshape(-1,1))))
        
        plt.plot(plotrange, prediction.data.numpy(), 'k.', label='NN prediction')
        plt.legend()
        plt.title('Test Loss=%.4f' % testloss)
        
        plt.savefig('prediction.jpg')
    