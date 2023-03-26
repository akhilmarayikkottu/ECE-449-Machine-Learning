import torch
import hw4_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    #pass
    d = X.size(dim=1)
    n = X.size(dim=0)
    
    temp =  torch.ones(Y.size())
    X1   = torch.cat((temp,X),1)
    w = torch.zeros((d+1,1))
    #print('size of w', w.size())
    i = 0
    while (i <num_iter):
        temp1 = torch.matmul(X1,w)-Y
        temp2 = torch.transpose(X1,0,1)
        temp3 = torch.matmul(temp2,temp1)
        w = w-lrate*(temp3)/n
        i = i+1
    return(w)

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    #pass
    n    = Y.size(dim=0) 
    temp = torch.ones(Y.size()).reshape([n,1])
    X1   = torch.cat((temp,X),1)
    INV  = torch.pinverse(X1)
    w    = torch.matmul(INV,Y)
    return(w)

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    #pass
    X_samp,Y_samp = utils.load_reg_data()
    w_samp = linear_normal(X_samp,Y_samp)
    X_temp = torch.linspace(-.5,4.5,50)
    X      = X_temp.reshape([50,1])
    temp   =  torch.ones(X.size())
    X1     = torch.cat((temp,X),1) 
    Y1     = torch.matmul(X1,w_samp)
    plt.scatter(X_samp,Y_samp,label='Data points',color='b',linewidth=3)
    plt.plot(X,Y1,'r',label='Linear reg. curve')
    plt.xlabel('X',fontsize =18)
    plt.ylabel('Y',fontsize =18)
    plt.legend(fontsize     =18)
    #plt.show()
    plt.savefig('LinReg.pdf')

# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    #pass
    d    = X.size(dim=1); n = X.size(dim=0)
    temp = torch.ones(Y.size())
    X1   = torch.cat((temp,X),1)
    for j in range (0,d):
        temp2 = X[:,j:]; temp3 = X[:,j].reshape([X.size(dim=0),1])
        temp4 = torch.mul(temp2,temp3)
        X1 = torch.cat((X1,temp4),1)
    w = torch.zeros((int(1+d+0.5*d*(d+1)),1))
    i = 0
    while (i <num_iter):
        temp1 = torch.matmul(X1,w)-Y
        temp2 = torch.transpose(X1,0,1)
        temp3 = torch.matmul(temp2,temp1)
        w = w-lrate*(temp3)/n
        i = i+1
    return(w)
    
    

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    #pass
    d    = X.size(dim=1)
    n    = Y.size(dim=0)
    temp = torch.ones(Y.size()).reshape([n,1])
    X1   = torch.cat((temp,X),1)
    for j in range (0,d):
        temp2 = X[:,j:]; temp3 = X[:,j].reshape([X.size(dim=0),1])
        temp4 = torch.mul(temp2,temp3)
        X1 = torch.cat((X1,temp4),1)
    INV  = torch.pinverse(X1)
    w    = torch.matmul(INV,Y)
    return(w)
    
    

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass
    nodes = 50
    X_samp,Y_samp = utils.load_reg_data()
    w_samp = poly_normal(X_samp,Y_samp)
    X_temp = torch.linspace(-.5,4.5,nodes).reshape([nodes,1])
    temp   = torch.ones(X_temp.size())
    X = torch.cat((temp,X_temp,X_temp**2),1)

    Y1     = torch.matmul(X,w_samp)
    plt.scatter(X_samp,Y_samp,label='Data points',color='b',linewidth=4)
    plt.plot(X_temp,Y1,'r',label='Poly reg. curve')
    plt.xlabel('X',fontsize =18)
    plt.ylabel('Y',fontsize =18)
    plt.legend(fontsize     =18)
    #plt.show()
    plt.savefig('PolyReg.pdf')
 

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    pass
    def PolyFunc(X):
        Xxor,Yxor = utils.load_xor_data()
        w  =  poly_normal(Xxor,Yxor)
        n  =  X.size(dim=0)
        X1 = X[:,0].reshape([n,1]); X2 = X[:,1].reshape([n,1])
        temp   = torch.ones(X1.size())
        X = torch.cat((temp,X1,X2,X1**2,X1*X2,X2**2),1)
        Y1     = torch.matmul(X,w)
        return(Y1)
    
    def LineFunc(X):
        Xxor,Yxor = utils.load_xor_data()
        w  =  linear_normal(Xxor,Yxor)
        n  =  X.size(dim=0)
        X1 = X[:,0].reshape([n,1]); X2 = X[:,1].reshape([n,1])
        temp   = torch.ones(X1.size())
        X = torch.cat((temp,X1,X2),1)
        Y1     = torch.matmul(X,w)
        return(Y1)

    utils.contour_plot_p(-2.0, 2.0, -2.0, 2.0, PolyFunc, ngrid = 33) 
    utils.contour_plot_l(-2.0, 2.0, -2.0, 2.0, LineFunc, ngrid = 33)

#plot_linear()
#plot_poly()
poly_xor()
