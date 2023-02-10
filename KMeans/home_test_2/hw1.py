import torch
import hw1_utils

def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """
    i = 0 
    if X is None:
        X, init_c = hw1_utils.load_data_2()  
    pass
    N = X.size(dim=1)
    c = init_c
    break_s = 0
    refTen = torch.ones(N)*100.0

    ## Plot the initial data and the initialized cluster centers
    iC1 = torch.tensor([[2],[2]])
    iC2 = torch.tensor([[2],[-2]])
    hw1_utils.vis_cluster(iC1,X,iC2,X)

    while (i < n_iters and break_s == 0 ):
        #distance from centroid 1
        dx_1   = X[0,:]-c[0,0].numpy()
        dy_1   = X[1,:]-c[1,0].numpy()
        D_1_sq = torch.square(dx_1)+torch.square(dy_1)

        #distance from centroid 2
        dx_2   = X[0,:]-c[0,1].numpy()
        dy_2   = X[1,:]-c[1,1].numpy()
        D_2_sq = torch.square(dx_2)+torch.square(dy_2)
          
        # difference between distance to centroid 2 and 1 for a data point
        D_21   = D_2_sq-D_1_sq 

        # Sortf assignment
        Pseudo_assig =  D_21/abs(D_21)

        # specific assignments to clusters
        r1 = (Pseudo_assig+1)/2
        r2 = (Pseudo_assig-1)/(-2)

        c_00 = torch.sum(r1*X[0,:])/torch.sum(r1) ; c_10 = torch.sum(r1*X[1,:])/torch.sum(r1)
        c_01 = torch.sum(r2*X[0,:])/torch.sum(r2) ; c_11 = torch.sum(r2*X[1,:])/torch.sum(r2)

        c = torch.tensor([[c_00, c_01],[c_10, c_11]])
        i = i+1
        if (torch.equal(refTen, Pseudo_assig)) :
            break_s = 1    
        refTen = Pseudo_assig

        ## For plotting and cost calculation
        
        Xl1  = torch.transpose((X[0,:]*r1)[(X[0,:]*r1).nonzero()],0,1)
        Yl1  = torch.transpose((X[1,:]*r1)[(X[1,:]*r1).nonzero()],0,1)
        X1   = torch.stack((Xl1,Yl1))
        Xl2  = torch.transpose((X[0,:]*r2)[(X[0,:]*r2).nonzero()],0,1)
        Yl2  = torch.transpose((X[1,:]*r2)[(X[1,:]*r2).nonzero()],0,1)
        X2   = torch.stack((Xl2,Yl2))
       
        # cost function
        cf = torch.sum(D_1_sq*r1)+torch.sum(D_2_sq*r2)
        print('Cost function ~~~~~>', cf/2)
         
        Cv1 = torch.tensor([[c_00],[c_10]])
        Cv2 = torch.tensor([[c_01],[c_11]])
        hw1_utils.vis_cluster(Cv1, X1, Cv2, X2)
        print('Number of iterations~~~~~~>', i)
        print('Cluster center ~~~~~~>', c)
    return (c)


k_means(None, None,500)
