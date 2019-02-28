# Simple Neural Network that learns a double well function. Graphviz is used to visualise the trajectory in weight space.
# Extend by a fractal-like function


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('agg') # different backend so we don't need tkinter
import matplotlib.pyplot as plt


def dblWell(x): # double well
    E0=1
    E1=0.05
    center_x = 5
    
    x = x-center_x
    return E0*(E1*x**4-x**2)

def fractalLike(x, L=10): # fractal-like Fourier series from Ann. Stat. 34, 1636
    coeffs = [.21, 1.25, .61, .25, .13, .10, 1.16, .18, .12, .23, .21, .19, .37, .99, .36, .02, .06, .08, .09, .04]
    series = 0.0
    for i,coeff in enumerate(coeffs):
        series += coeff*np.sin(i*2.0*np.pi*x/L)
    return 2.0*series


class Net(torch.nn.Module): # from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/301_regression.py
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # single hidden layer with n_hidden output features
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


if __name__ == "__main__":
    # sample from function
    N       = 30         # how many total sampled points
    N_train = int(0.8*N) # how many training points
    
    L = 10 # sample points between 0 and this value
    X = L*np.random.rand(N)
    
    samplingFunction = dblWell
    y = samplingFunction(X) # choose which function to sample from
    
    X_train = X[:N_train] # training data
    y_train = y[:N_train]
    X_test = X[N_train:]  # validation data
    y_test = y[N_train:]

    
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
        
        net = Net(n_features, n_hidden=10, n_output=1)     # define the network
        x = torch.from_numpy(X_train).float().view(N_train, n_features)
        y = torch.from_numpy(y_train).float()
        
        optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        
        for t in range(200):
            prediction = net(x)     # input x and predict based on x
            loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            if t % 5 == 0:
                # show learning process
                print('Loss=%.4f' % loss.data.numpy())
                if (progress_plots):
                    plt.cla()
                    plt.scatter(x.data.numpy(), y.data.numpy())
                    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
                    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color':  'red'})
                    plt.savefig('tmp'+str(t)+'.jpg')
        
        plotrange = np.linspace(0, L, num=120)
        plt.plot(plotrange, samplingFunction(plotrange), label='true fct.')
        plt.plot(X_train, y_train, '.', label='training data')
        plt.plot(X_test, y_test, 'r.', label='test data')
        plt.ylim(-10,15)
        
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', label='NN prediction')
        plt.legend()
        plt.text(0.5, 10, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 12, 'color':  'red'})
        
        plt.savefig('prediction.jpg')
    