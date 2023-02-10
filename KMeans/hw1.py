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
    print('nuumber of iteration:' , n_iters)
    i = 0 
    if X is None:
        X, init_c = hw1_utils.load_data()  
    pass
    N = X.size(dim=1)
    c = init_c

    while (i < n_iters):
        #distance from centroid 1
        dx_1   = X[0,:]-c[0,0].numpy()
        dy_1   = X[1,:]-c[1,0].numpy()
        D_1_sq = torch.square(dx_1)+torch.square(dy_1)
#        print (D_1_sq)

        #distance from centroid 2
        dx_2   = X[0,:]-c[0,1].numpy()
        dy_2   = X[1,:]-c[1,1].numpy()
        D_2_sq = torch.square(dx_2)+torch.square(dy_2)
#        print (D_2_sq)
          
        # difference between distance to centroid 2 and 1 for a data point
        D_21   = D_2_sq-D_1_sq 
#        print(D_21)

        # Sortf assignment
        Pseudo_assig =  D_21/abs(D_21)
#        print(Pseudo_assig)

        # specific assignments to clusters
        r1 = (Pseudo_assig+1)/2
        r2 = (Pseudo_assig-1)/(-2)
#        print(r1)
#        print(r2)

        c_00 = torch.sum(r1*X[0,:])/torch.sum(r1) ; c_10 = torch.sum(r1*X[1,:])/torch.sum(r1)
        c_01 = torch.sum(r2*X[0,:])/torch.sum(r2) ; c_11 = torch.sum(r2*X[1,:])/torch.sum(r2)
#        print(c_00)

        c = torch.tensor([[c_00, c_01],[c_10, c_11]])
        i = i+1
    print('Number of iterations~~~~~~>', i)
    print('Cluster center ~~~~~~>', c)
    #hw1_utils.vis_cluster(X, init_c[:,0], X, init_c[:,1])
    return (c)
#    print(X[0,:])
#    print(X[0,:]-init_c[0,0].numpy())


k_means(None, None,50)
