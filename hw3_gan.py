import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw3_utils import BASE_URL, download, GANDataset


class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # TODO: implement layers here
        self.conv1 = nn.Conv2d(1, 2, kernel_size =3 , stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.maxp3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv4 = nn.Conv2d(2, 4, kernel_size =3 , stride=1, padding=1, bias=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxp6 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv7 = nn.Conv2d(4, 8, kernel_size =3 , stride=1, padding=0, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.line9 = nn.Linear(200,1,bias=True)

        self._weight_init()
    

    def _weight_init(self):
        # TODO: implement weight initialization here
        for layer in self.children():
            if type(layer) == nn.Linear:
                nn.init.kaiming_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0)
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0)

        pass        

    def forward(self, x):
        # TODO: complete forward function
        out = x
        #print('size of input', out.size())
        out = self.conv1(out)
        #print('size of conv1', out.size())
        out = self.relu2(out)
        out = self.maxp3(out)
        out = self.conv4(out)
        out = self.relu5(out)
        out = self.maxp6(out)
        out = self.conv7(out)
        out = self.relu8(out)
        out = nn.Flatten()(out)
        out = self.line9(out)
        return(out)


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()
        # TODO: implement layers here
        self.line1  = torch.nn.Linear(zdim, 1568,bias=True)
        self.lrelu2 = torch.nn.LeakyReLU(0.2)
        self.reshap = torch.nn.Unflatten(1,(32,7,7))
        self.upsam3 = torch.nn.Upsample(scale_factor=2)
        self.conv4  = torch.nn.Conv2d(32, 16, kernel_size =3 , stride=1, padding=1, bias=True)
        self.lrelu5 = torch.nn.LeakyReLU(0.2)
        self.upsam6 = torch.nn.Upsample(scale_factor=2)
        self.conv7  = torch.nn.Conv2d(16, 8, kernel_size =3 , stride=1, padding=1, bias=True)
        self.lrelu8 = torch.nn.LeakyReLU(0.2)
        self.conv9  = torch.nn.Conv2d(8, 1, kernel_size =3 , stride=1, padding=1, bias=True)

        self._weight_init()
    
#    @staticmethod
    def _weight_init(self):
        # TODO: implement weight initialization here
        for layer in self.children():
            if type(layer) == nn.Linear:
                nn.init.kaiming_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0)
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0)
 
        pass    

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        #pass
        out = z
        #print('size of z', out.size())
        out = self.line1(out)
        #print('out after line1', out.size())
        out = self.lrelu2(out)
        batch_sz = out.size(dim=0)
        batch_sz = int(batch_sz)
        #print('batch size', batch_sz)
        out = torch.reshape(out,(batch_sz,32,7,7))   
        out = self.upsam3(out)
        out = self.conv4(out)
        out = self.lrelu5(out)
        out = self.upsam6(out)
        out = self.conv7(out)
        out = self.lrelu8(out)
        out = self.conv9(out)
        out = torch.sigmoid(out)
        return(out)

class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        
        # Real data predictions
        #print('size of batch', batch_data.size())
        prediction_real = self.disc(batch_data)
        #print('weight for predictor', prediction_real.size())
        size_real       = batch_data.size(0)
        target_real     = torch.ones(size_real, 1)
        #print('size of real target', target_real.size())
        loss1 = nn.BCEWithLogitsLoss()
        real_error = loss1(prediction_real, target_real)

        # Fake data 
        generated_data = self.gen(z)
        size_fake      = generated_data.size(0)
        target_fake    = torch.zeros(size_fake, 1)
        predicted_fake = self.disc(generated_data)
        #predicted_fake = torch.ones(predicted_fake.size())-predicted_fake
        loss2 = nn.BCEWithLogitsLoss()
        fake_error = loss2(predicted_fake, target_fake)
        
        return(0.5*(real_error+fake_error))
        pass

    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.
        Compute -\sum_z\log{D(G(z))} instead of \sum_z\log{1-D(G(z))}
        
        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        generated_data = self.gen(z)
        size           = generated_data.size(0)
        target         = torch.ones(size, 1)
        predicted_data = self.disc(generated_data)
        loss = nn.BCEWithLogitsLoss()
        error = loss(predicted_data, target)
        return(error)
        pass

    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()
