"""
Training Procedure:

Input: image projected onto sphere
Layers: SO(3) Conv, SO(3) activations, SO(2) conv
Output: prob distribution over SO(3)

Take all poses with probability above certain threshold, look at projections of 
ground truth volume at those poses, loss is sum of MSEs of those
PDB to overfit to: 6BDF (cylinder)

Inference Procedure:
For each input image, generate list of poses
Determinstic reconstruction
"""
import argparse
import e3nn
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import model as model
import predictor as predictor
from projection import specific_projection_pda
import so3_utils

import time
from tqdm import tqdm

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

        with open('/home/dnori/cryo_reconstruction/ground_truth/all.pickle', 'rb') as pickle_result:
            self.ground_truth = pickle.load(pickle_result)
        
        if not self.ground_truth:
            raise NotImplementedError("Missing ground truth data")
        
        for i in range(len(self.ground_truth)):
            self.ground_truth[i] = torch.from_numpy(self.ground_truth[i]).cuda()
    
    def forward(self, fmap, probabilities):
        loss = 0
        for i in range(probabilities.shape[1]):
            p = probabilities[0,i]
            gt = self.ground_truth[i]
            mse = ((gt - fmap)*(gt-fmap)).mean()
            loss += p * p * mse
        return loss

class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        lmax=args.lmax
        s2_fdim=args.sphere_fdim
        so3_fdim_1 = 64
        so3_fdim_2 = 16
        s2_kernel_grid = so3_utils.s2_healpix_grid(max_beta=np.inf, rec_level=2)
        so3_kernel_grid = so3_utils.so3_near_identity_grid()

        so3_kernel_grid = so3_utils.so3_near_identity_grid()
        self.layers = nn.Sequential(
            model.S2Convolution(s2_fdim, so3_fdim_1, lmax, s2_kernel_grid),
            e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10),
            model.SO3Convolution(so3_fdim_1, so3_fdim_2, lmax, so3_kernel_grid),
            e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10),
            model.SO3Convolution(so3_fdim_2, 1, lmax, so3_kernel_grid)
        ).cuda()

        output_xyx = so3_utils.so3_healpix_grid(rec_level=2).cuda() #rec_level is resolution
        self.sphere_grid = output_xyx
        self.eval_wigners = so3_utils.flat_wigner(lmax, *output_xyx).transpose(0,1).cuda() # for probability calculation

    def forward(self, x):
        harmonics = self.layers(x)
        probabilities = self.compute_probabilities(harmonics) #[1,4608]
        return probabilities
    
    def compute_probabilities(self, harmonics):
        ''' compute probabilities over grid resulting from SO(3) group convolution'''
        probs = torch.matmul(harmonics, self.eval_wigners).squeeze(1)
        return nn.Softmax(dim=1)(probs)

    def position_to_pose(self, indices):
        # tensor of indices (N,)
        # returns tensor of size (N,3) where each index has alpha, beta, gamma
        # total depends on rec_level (resolution = 2, 4608 points, bin width = 15 degrees)
        return self.sphere_grid[:,indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_shape', type=int, default=512)
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--proj_per_img', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--sphere_fdim', type=int, default=128)
    parser.add_argument('--lmax', type=int, default=4)
    parser.add_argument('--harmonic_coefs', type=int, default=25)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    with open('projections.pickle', 'rb') as pickle_file:
        projection_dict = pickle.load(pickle_file)

    with open('pdas.pickle', 'rb') as pickle_file:
        pda_dict = pickle.load(pickle_file)

    shape = (args.img_shape, args.img_shape)

    projector = model.Image2SphereProjector(fmap_shape=(1,)+shape, sphere_fdim=args.sphere_fdim, lmax=args.lmax)
    input = torch.zeros((len(projection_dict.items()), args.proj_per_img, args.sphere_fdim, args.harmonic_coefs))
    fmaps = torch.zeros((len(projection_dict.items()), args.proj_per_img, args.img_shape, args.img_shape))
    true_pdas = [] # list of pdas where each pda is (coords, densities)

    for i in range(len(projection_dict.items())):
        pdb_id, projections = list(projection_dict.items())[i]
        true_pdas.append(pda_dict[pdb_id])
        for j in range(len(projections)):
            fmap = projections[j]
            proj = projector(torch.from_numpy(fmap[np.newaxis, np.newaxis, : ,:]).float())[0,:,:]
            input[i, j, :, :] = proj
            fmaps[i, j, :, :] = torch.from_numpy(fmap)

    # input is shape (# of proteins, # of projections per protein, sphere_fdim, # of spherical harmonic coefs)
    # add train test split (perhaps sequence similarity split, start with random)

    # starting with very simple case - training with only one protein
    # input: a projection

    model = Encoder(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    criterion = CustomLoss()

    for e in range(args.epochs):
        start_time = time.time()
        model.train()
        tot_loss = 0
        for p in range(input.shape[0]):
            for i in tqdm(range(10)): #input.shape[1]
                # print(f"\t\tprojection {i} epoch {e}")
                probabilities = model.forward(input[p,i,:].cuda())
                loss = criterion(fmaps[p,i,:].cuda(), probabilities.cuda())
                loss.backward(retain_graph=True)
                tot_loss += loss

        optimizer.step()
        optimizer.zero_grad()
        print(f"Avg loss epoch {e}: {tot_loss/10}.")
                
        epoch_time = time.time() - start_time
        print(f"Epoch {e} running time: {epoch_time}.")

    
    

    

    