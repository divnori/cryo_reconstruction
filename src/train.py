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

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        lmax=4
        s2_fdim=1
        so3_fdim=16
        s2_kernel_grid = so3_utils.s2_healpix_grid(max_beta=np.inf, rec_level=1)
        so3_kernel_grid = so3_utils.so3_near_identity_grid()

        so3_kernel_grid = so3_utils.so3_near_identity_grid()
        self.layers = nn.Sequential(
            model.S2Convolution(s2_fdim, so3_fdim, lmax, s2_kernel_grid),
            e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10),
            model.SO3Convolution(so3_fdim, 1, lmax, so3_kernel_grid)
        ).cuda()

        output_xyx = so3_utils.so3_healpix_grid(rec_level=2).cuda() #rec_level is resolution
        self.sphere_grid = output_xyx
        self.eval_wigners = so3_utils.flat_wigner(lmax, *output_xyx).transpose(0,1).cuda() # for probability calculation

    def forward(self, x, fmap, true_pda):
        harmonics = self.layers(x)
        print("computed harmonics")
        probabilities = self.compute_probabilities(harmonics)
        fifth_highest_val = torch.topk(probabilities, 5)[0].cpu().detach().numpy()[0,4]
        probabilities = torch.where(probabilities < fifth_highest_val, 0, probabilities)
        high_prob_indices = probabilities.nonzero() #taking all nonzero probabilities as tensor of (N, 2)
        poses = self.position_to_pose(high_prob_indices.T[1]) # poses shape (3, # of nonzero probability poses)
        print("got poses")
        projections = specific_projection_pda(poses, true_pda)
        print("got projections")
        loss = 0
        for proj in projections:
            loss += F.mse_loss(torch.from_numpy(proj), torch.from_numpy(fmap))
        return harmonics, probabilities, loss
    
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    with open('projections.pickle', 'rb') as pickle_file:
        projection_dict = pickle.load(pickle_file)

    with open('pdas.pickle', 'rb') as pickle_file:
        pda_dict = pickle.load(pickle_file)

    shape = (args.img_shape, args.img_shape)

    projector = model.Image2SphereProjector(fmap_shape=(1,)+shape, sphere_fdim=1, lmax=4)
    input = torch.zeros((len(projection_dict.items()), args.proj_per_img, 25))
    true_pdas = [] # list of pdas where each pda is (coords, densities)

    for i in range(len(projection_dict.items())):
        pdb_id, projections = list(projection_dict.items())[i]
        true_pdas.append(pda_dict[pdb_id])
        for j in range(len(projections)):
            fmap = projections[j]
            proj = projector(torch.from_numpy(fmap[np.newaxis, np.newaxis, : ,:]).float())[0,0,:]
            input[i, j, :] = proj

    # input is shape (# of proteins, # of projections per protein, # of spherical harmonic coefs)
    # add train test split (perhaps sequence similarity split, start with random)

    # starting with very simple case - training with only one protein
    # input: a projection

    model = Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    for e in range(args.epochs):
        model.train()
        for p in range(input.shape[0]):
            for i in range(1): #for i in range(input.shape[1])
                print(f"projection {i} epoch {e}")
                optimizer.zero_grad()
                harmonics, probabilities, loss = model.forward(input[p,i,:].cuda(), fmap, true_pdas[p])
                print(f"loss {loss}")

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

    
    

    

    