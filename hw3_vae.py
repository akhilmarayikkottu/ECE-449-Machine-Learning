import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from hw3_utils import array_to_image, concat_images, batch_indices, load_mnist

# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # fc1: a fully connected layer with 500 hidden units. 
        # fc2: a fully connected layer with 500 hidden units.
        self.fc1 = nn.Linear(latent_dimension,hidden_units)
        self.fc2 = nn.Linear(hidden_units, 784)

    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]

        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.
        out = self.fc1(z)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        p = out
        return p


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches
        self.e_path = args.e_path
        self.d_path = args.d_path
        
        #print('self.dpath', self.d_path)
        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()

        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if self.resume_training:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]

        #print('sigma', sigma_square)
        #temp   = torch.mul(torch.ones(sigma_square.size()),torch.normal(torch.tensor(0), torch.tensor(1)))
        #print('mu in sample diagn gaussian', mu.size())
        temp = torch.normal(mean=0, std= 1.0, size = (mu.size(dim=0),mu.size(dim=1)))
        sample = mu+torch.mul(temp,torch.sqrt(sigma_square))
        return sample

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension], type should be torch.float32

        x=torch.bernoulli(p) 
        return x


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagonal gaussian [batch_size]
        
        t1 = torch.div(torch.mul((z-mu),(z-mu)),2*sigma_square)
        t2 = torch.log((2*3.141592654)**(sigma_square.size(dim=1))*torch.prod(sigma_square,dim=1,dtype=float))
        logprob = -0.5*t2-torch.sum(t1,dim=1)
        return logprob

    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]
        
        #print('SIZE', p.size(),x.size())
        t1 = torch.mul(torch.log(p),x)
        t2 = torch.mul(torch.log(torch.ones(p.size())-p),(torch.ones(x.size())-x))
        logprob = torch.sum(t1,1)+torch.sum(t2,1)
        return logprob
    
    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        zs = self.sample_diagonal_gaussian(mu, sigma_square)
        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu:
        #   sigma_square: parameters of q(z|x) [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        # log_p_z(z) log probability of z under prior
        z_mu = torch.FloatTensor([0]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        z_sigma = torch.FloatTensor([1]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        # log_p(x|z) - conditional probability of data given latents.
        log_p = self.logpdf_bernoulli(x, p)
        
        elbo = torch.mean(log_p+log_p_z-log_q)
        return elbo


    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = self.num_epoches * num_batches
        
        for i in range(num_iters):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            #print('size of p',p.size())
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()

            if i%100 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

        # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    def visualize_data_space(self):
        mu = torch.zeros(10,2)
        sigma_square = torch.ones(10,2)
        z = self.sample_diagonal_gaussian(mu,sigma_square)
        p_xz = self.decoder(z)
        
        images1 = []; images2 =[]
        for i in range (0,10):
            p_xz_array = p_xz[i,:].detach().numpy()
            reshaped_array = array_to_image(p_xz_array)
            images1.append(reshaped_array)
            monteC = np.random.uniform(0,1,784)
            samp   = np.zeros(784)
            for j in range(0,784):
                if (p_xz_array[j] >= monteC[j]):
                    samp[j] = 1
            reshaped_array = array_to_image(samp)
            images1.append(reshaped_array)
        

        images = images1+images2
        conc = concat_images(images1, 2, 10, padding = 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(conc,cmap='gray')
        plt.show()
        
        
        
    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector 
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot 
    # by the class label for the input data. Each point in the plot is colored by the class label for 
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though 
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        #print('inside visualize_latent_space')
        
        # TODO: Encode the training data self.train_images
        #print('training set', self.train_images.size())
        mu, sigma_square = self.encoder(self.train_images)
        #print('size of mu', mu.size())
        #print('training labels', self.train_labels.size()) 
        # TODO: Take the mean vector of each encoding
        

        # TODO: Plot these mean vectors in the latent space with a scatter
        # Colour each point depending on the class label
        #mu_array  = mu.detach().numpy()
        mu_x_0 = []; mu_y_0 = []
        mu_x_1 = []; mu_y_1 = []
        mu_x_2 = []; mu_y_2 = []
        mu_x_3 = []; mu_y_3 = []
        mu_x_4 = []; mu_y_4 = []
        mu_x_5 = []; mu_y_5 = []
        mu_x_6 = []; mu_y_6 = []
        mu_x_7 = []; mu_y_7 = []
        mu_x_8 = []; mu_y_8 = []
        mu_x_9 = []; mu_y_9 = []

        for i in range(0,9999):
            x1 = float(mu[i][0])
            y1 = float(mu[i][1])
            temp = torch.tensor([0,1,2,3,4,5,6,7,8,9])
            label = int(torch.sum(torch.mul(self.train_labels[i][:],temp)))
            if (label == 0):
                mu_x_0.append(x1)
                mu_y_0.append(y1)
            if (label == 1):
                mu_x_1.append(x1)
                mu_y_1.append(y1)
            if (label == 2):
                mu_x_2.append(x1)
                mu_y_2.append(y1)
            if (label == 3):
                mu_x_3.append(x1)
                mu_y_3.append(y1)
            if (label == 4):
                mu_x_4.append(x1)
                mu_y_4.append(y1)
            if (label == 5):
                mu_x_5.append(x1)
                mu_y_5.append(y1)
            if (label == 6):
                mu_x_6.append(x1)
                mu_y_6.append(y1)
            if (label == 7):
                mu_x_7.append(x1)
                mu_y_7.append(y1)
            if (label == 8):
                mu_x_8.append(x1)
                mu_y_8.append(y1)
            if (label == 9):
                mu_x_9.append(x1)
                mu_y_9.append(y1)
    
        plt.scatter( mu_x_0,mu_y_0 ,c ='r',label='0',marker ='x',alpha=.5)
        plt.scatter( mu_x_1,mu_y_1 ,c ='b',label='1',marker ='x')
        plt.scatter( mu_x_2,mu_y_2 ,c ='g',label='2',marker ='x')
        plt.scatter( mu_x_3,mu_y_3 ,c ='k',label='3',marker ='x')
        plt.scatter( mu_x_4,mu_y_4 ,c ='orange',label='4',marker ='x')
        plt.scatter( mu_x_5,mu_y_5 ,c ='magenta',label='5',marker ='x')
        plt.scatter( mu_x_6,mu_y_6 ,c ='yellow',label='6',marker ='x')
        plt.scatter( mu_x_7,mu_y_7 ,c ='pink',label='7',marker ='x')
        plt.scatter( mu_x_8,mu_y_8 ,c ='cyan',label='8',marker ='x')
        plt.scatter( mu_x_9,mu_y_9 ,c ='grey',label='9',marker ='x')
        plt.xlabel('$z_1$',fontsize = 22)
        plt.ylabel('$z_2$',fontsize = 22)
        plt.legend()
        plt.show()
 

        # TODO: Save the generated figure and include it in your report
        


    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):
        # TODO: Sample 3 pairs of data with different classes
        print('size of train data', self.train_images[1,:].size())
        i = 0
        pair = []
        while (i < 3) :
            print ('i',i)
            P1 = int(0+np.random.uniform(0,1,1)*9999)
            P2 = int(0+np.random.uniform(0,1,1)*9999)
            temp = torch.tensor([0,1,2,3,4,5,6,7,8,9])
            l1 = int(torch.sum(torch.mul(self.train_labels[P1,:],temp)))
            l2 = int(torch.sum(torch.mul(self.train_labels[P2,:],temp)))
            if (l1 != l2):
                pair.append([P1,P2])
                print('labels', l1,l2)
                i = i+1
        print(pair)        
        # TODO: Encode the data in each pair, and take the mean vectors
        # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
        # TODO: Along the interpolation, plot the distributions p(x|z_α)
        # Concatenate these plots into one figure
        images = []
        #for i in range (0,3):
        #    mu1,sigma_sq1 = self.encoder(self.train_images[pair[i][0],:])
        #    mu2,sigma_sq2 = self.encoder(self.train_images[pair[i][1],:])
        #    print('label in loop',self.train_labels[pair[i][0],:],self.train_labels[pair[i][1],:])
        for j in range (0,11):
            for i in range (0,3):
                mu1,sigma_sq1 = self.encoder(self.train_images[pair[i][0],:])
                mu2,sigma_sq2 = self.encoder(self.train_images[pair[i][1],:])

                mu_alp  = self.interpolate_mu(mu1, mu2, alpha = 0.1*j)
                mu_alp = torch.reshape(mu_alp,(1,2))
                var_alp = torch.ones(mu_alp.size())
                z = self.sample_diagonal_gaussian(mu_alp,var_alp)
                p_xz = self.decoder(z)
                p_xz_array = p_xz.detach().numpy()
                reshaped_array = array_to_image(p_xz_array)
                images.append(reshaped_array)

        print(np.shape(np.array(images)))
        conc = concat_images(images, 3, 11, padding = 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(conc,cmap='gray')
        plt.show()
 
      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


def main():
    
    # read the function arguments
    args = parse_args()
    
    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # train the model 
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()



################## RUN THE PROGRAM #################################
main()
