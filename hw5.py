import hw5_utils as utils
#import hw4 as hw4
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc as matplotlibrc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import  MultipleLocator
matplotlibrc('text',usetex=True)
matplotlibrc('font',family='serif')



def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    pass
    n = y_train.size(dim=0)
    K = torch.zeros([n,n])
    for i in range(0,n):
        for j in range(0,n):
            K[i,j] = kernel(x_train[i,:],x_train[j,:])
    B =  torch.outer(y_train,y_train)*kernel(x_train,x_train)
    alpha =  torch.zeros(n, requires_grad=True)
    for i in range (0,num_iters):
        loss = torch.sum(0.5*torch.outer(alpha*y_train,alpha*y_train)*K)-torch.sum(alpha)
        loss.backward()
        with torch.no_grad():
            alpha -= lr*alpha.grad
            alpha.clamp_(min=0,max=c)
            alpha.grad.zero_()
    return(alpha.detach())

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    pass
    n = alpha.size(dim=0);m = x_test.size(dim=0)
    K = torch.zeros([m,n])
    for i in range(0,m):
        for j in range(0,n):
            K[i,j] = kernel(x_train[j,:],x_test[i,:])

    y_prediction = torch.matmul(K,alpha*y_train)
    out  = y_prediction
    return(out)


def logistic(X, Y, lrate=.01, num_iter=1000):
    pass
    d    = X.size(dim=1); n    = Y.size(dim=0)
    temp = torch.ones(Y.size()).reshape([n,1])
    X1   = torch.cat((temp,X),1);w    = torch.zeros((d+1,1))

    for i in range(0,num_iter):
        Numer = torch.zeros((d+1,1))
        for j in range (0,n):
            X_temp = X1[j,:]; Y_temp =  Y[j]
            Z_temp = (X_temp*Y_temp).reshape([d+1,1])
            Numer  = Numer+Z_temp/(1+torch.exp(torch.matmul(w.reshape([1,d+1]),Z_temp)))
        w = w+lrate*Numer/(n)            
     
    return(w)


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass
    X,Y = utils.load_logistic_data() 
    w = logistic(X,Y, lrate=.01, num_iter=100000)
    X_1 = torch.linspace(-6.0,6.0,50)
    X_2 = (-w[0,0]-w[1,0]*X_1)/(w[2,0])
    plt.plot(X_1,X_2,'g',label='Logistic reg. curve',linewidth=3)

    w_linear = hw4.linear_normal(X,Y)
    X_l      = X_1.reshape([50,1])
    temp   =  torch.ones(X_l.size())
    X1     = torch.cat((temp,X_l),1)
    X1     = torch.cat((X1,X_l),1)
    Y1     = torch.matmul(X1,w_linear)
    plt.plot(X_l,Y1,'k',label='Linear reg. curve',linewidth=3)
    plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',marker='x')
    plt.legend(fontsize     =18)
    plt.xlabel('$x_1$',fontsize =18); plt.ylabel('$x_2$',fontsize =18)
    plt.xlim(-5.5,5.5); plt.ylim(-5.5,5.5)
    plt.tick_params(axis='both',which='both',labelsize=18, direction = 'in')
    #plt.show()
    plt.savefig('LogReg.pdf')


#logistic_vs_ols()  

##X_xor, Y_xor = utils.xor_data()
##LR = 0.1; NUM = 10000

##alpha_opt = svm_solver(X_xor, Y_xor, LR, NUM,
##               kernel=utils.poly(degree=2), c=None)

##def preTest( x_test):
##    pass
##    alpha = alpha_opt;x_train = X_xor;y_train = Y_xor
##    kernel = utils.poly(degree=2)
##    n = alpha.size(dim=0);m = x_test.size(dim=0)
##    K = torch.zeros([m,n])
##    for i in range(0,m):
##        for j in range(0,n):
##            K[i,j] = kernel(x_train[j,:],x_test[i,:])
##
##    y_prediction = torch.matmul(K,alpha*y_train)
##    out  = y_prediction
##    return(out)

##utils.svm_contour(preTest, xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33)
