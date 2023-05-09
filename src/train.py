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
import numpy as np
import pickle
import torch
import torch.nn as nn
import model as model
import predictor as predictor
import so3_utils
import e3nn

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        lmax=4
        s2_fdim=1
        so3_fdim=16
        s2_kernel_grid = so3_utils.s2_healpix_grid(max_beta=np.inf, rec_level=1)
        print("defined s2")
        so3_kernel_grid = so3_utils.so3_near_identity_grid()
        print("defined so3")

        so3_kernel_grid = so3_utils.so3_near_identity_grid()
        print("defined so3")
        self.layers = nn.Sequential(
            model.S2Convolution(s2_fdim, so3_fdim, lmax, s2_kernel_grid),
            e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10),
            model.SO3Convolution(so3_fdim, 1, lmax, so3_kernel_grid)
        ).cuda()
        print("defined layers")

        output_xyx = so3_utils.so3_healpix_grid(rec_level=2).cuda() #rec_level is resolution
        self.eval_wigners = so3_utils.flat_wigner(lmax, *output_xyx).transpose(0,1).cuda()

    def forward(self, x):
        return self.layers(x)
    
    @torch.no_grad()
    def compute_probabilities(self, harmonics):
        ''' compute probabilities over grid resulting from SO(3) group convolution'''
        probs = torch.matmul(harmonics, self.eval_wigners).squeeze(1)

        return nn.Softmax(dim=1)(probs)

if __name__ == "__main__":
    with open('projections.pickle', 'rb') as pickle_file:
        projection_dict = pickle.load(pickle_file)

    shape = (512, 512)

    projector = model.Image2SphereProjector(fmap_shape=(1,)+shape, sphere_fdim=1, lmax=4)
    input = torch.zeros((len(projection_dict.items()), 50, 25))
    for i in range(len(projection_dict.items())):
        pdb_id, projections = list(projection_dict.items())[i]
        for j in range(len(projections)):
            fmap = projections[j]
            proj = projector(torch.from_numpy(fmap[np.newaxis, np.newaxis, : ,:]).float())[0,0,:]
            input[i, j, :] = proj

    # input is shape (# of proteins, # of projections per protein, # of spherical harmonic coefs)

    

    encoder = Encoder()

    



    harmonics = encoder.forward(input[0,0,:].cuda())
    probabilities = encoder.compute_probabilities(harmonics)
    print(probabilities.shape)



    
    

    

    