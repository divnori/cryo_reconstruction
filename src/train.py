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
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from binary_label_metrics import BinaryLabelMetrics
import model as model
import so3_utils

import time
from tqdm import tqdm


def generate_bin_mask(img_shape = 512, proj_per_protein = 500):
    """
    Used to generate ground truth binary masks for training.

    ## Parameters
    img_shape: int
        width/height of sample projections
    proj_per_img: int
        number of projections per protein

    ## Returns
    bin_mask: torch.Tensor
        Ground-truth binary mask used to compute loss during training/testing. 
    """

    with open('../data/projections.pickle', 'rb') as pickle_file:
        projection_dict = pickle.load(pickle_file)

    fmaps = torch.zeros((len(projection_dict.items()), proj_per_protein, img_shape, img_shape))

    for i in range(len(projection_dict.items())):
        pdb_id, projections = list(projection_dict.items())[i]
        for j in range(proj_per_protein):
            fmap = projections[j]
            fmaps[i, j, :, :] = torch.from_numpy(fmap)

    with open('../ground_truth/all.pickle', 'rb') as pickle_result:
        ground_truth = pickle.load(pickle_result)
    
    if not ground_truth:
        raise NotImplementedError("Missing ground truth data")
    
    for i in range(len(ground_truth)):
        ground_truth[i] = torch.from_numpy(ground_truth[i])
    
    mses = torch.zeros((fmaps.shape[0], fmaps.shape[1], len(ground_truth)))
    
    for i in range(mses.shape[0]):
        for j in range(mses.shape[1]):
            fmap = fmaps[i,j,:,:]
            for k in range(mses.shape[2]):
                gt = ground_truth[k]
                mses[i,j,k] = ((gt - fmap)*(gt-fmap)).mean()

    prob = torch.exp(-.0001*mses*mses)
    threshold_probs = np.repeat(np.expand_dims(np.expand_dims(np.array([torch.max(prob[0,i,:])-0.05 for i in range(fmaps.shape[1])]), axis=1).T, axis=2), self.prob.shape[2], axis=2)
    bin_mask = torch.where(prob > torch.from_numpy(threshold_probs), 1, 0)

    return bin_mask

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

        with open("bin_mask.pickle", "rb") as f:
            self.bin_mask = pickle.load(f)
            self.bin_mask = self.bin_mask.cuda()
    
    def forward(self, fmap_idx, probabilities):
        p = probabilities[0].float()
        loss = torch.nn.BCELoss()
        return  loss(p, self.bin_mask[fmap_idx[0],fmap_idx[1]].float())

class Encoder(nn.Module):
    """
    CryoSphere encoder model, which takes a 2D projection as input 
    and returns a probability map over SO(3) poses.
    """
    def __init__(self, args):
        super().__init__()
        self.blm = BinaryLabelMetrics()

        self.featurizer = nn.Sequential(
            nn.Conv2d(1, 3, 5, 2),
            nn.ReLU(),
            nn.Conv2d(3, 5, 5, 2)
        ).cuda()

        lmax=args.lmax
        s2_fdim=args.sphere_fdim
        so3_fdim_1 = 64
        s2_kernel_grid = so3_utils.s2_healpix_grid(max_beta=np.inf, rec_level=2)
        so3_kernel_grid = so3_utils.so3_near_identity_grid()

        so3_kernel_grid = so3_utils.so3_near_identity_grid()
        self.layers = nn.Sequential(
            model.S2Convolution(s2_fdim, so3_fdim_1, lmax, s2_kernel_grid),
            e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10),
            nn.BatchNorm1d(165),
            model.SO3Convolution(so3_fdim_1, 1, lmax, so3_kernel_grid),
        ).cuda()

        self.projector = projector = model.Image2SphereProjector(fmap_shape=(5,125, 125), sphere_fdim=args.sphere_fdim, lmax=args.lmax)

        output_xyx = so3_utils.so3_healpix_grid(rec_level=2).cuda() #rec_level is resolution
        self.sphere_grid = output_xyx
        self.eval_wigners = so3_utils.flat_wigner(lmax, *output_xyx).transpose(0,1).cuda() # for probability calculation

    def forward(self, x):
        # x is 2D orthographic projection
        
        # Initial 2d convolutional layers for feature extraction 
        feat = self.featurizer(x.cuda())

        # project onto half-sphere
        proj = self.projector(feat[0].cpu())[0,:,:].cuda()

        # spherical harmonic layers
        harmonics = self.layers(proj)

        probabilities = self.compute_probabilities(harmonics) #[1,4608]
        return probabilities
    
    def compute_probabilities(self, harmonics):
        ''' compute probabilities over grid resulting from SO(3) group convolution'''
        probs = torch.matmul(harmonics, self.eval_wigners).squeeze(1)
        return nn.Sigmoid()(probs)

    def position_to_pose(self, indices):
        # tensor of indices (N,)
        # returns tensor of size (N,3) where each index has alpha, beta, gamma
        # total depends on rec_level (resolution = 2, 4608 points, bin width = 15 degrees)
        return self.sphere_grid[:,indices]

# Main training loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_shape', type=int, default=512)
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--proj_per_img', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=201)
    parser.add_argument('--sphere_fdim', type=int, default=64)
    parser.add_argument('--lmax', type=int, default=4)
    parser.add_argument('--harmonic_coefs', type=int, default=25)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--num_train_img', type=float, default=400)
    parser.add_argument('--experiment_path', type=str, default='experiments/experiment_400images')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    with open('../data/projections.pickle', 'rb') as pickle_file:
        projection_dict = pickle.load(pickle_file)

    with open('../data/pda.pickle', 'rb') as pickle_file:
        pda_dict = pickle.load(pickle_file)

    shape = (args.img_shape, args.img_shape)

    fmaps = torch.zeros((len(projection_dict.items()), args.proj_per_img, args.img_shape, args.img_shape))

    for i in range(len(projection_dict.items())):
        pdb_id, projections = list(projection_dict.items())[i]
        for j in range(args.proj_per_img):
            fmap = projections[j]
            fmaps[i, j, :, :] = torch.from_numpy(fmap)

    criterion = CustomLoss()
    model = Encoder(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for e in range(args.epochs):
        optimizer.zero_grad()
        start_time = time.time()
        model.train()
        tot_loss = 0
        true_vals = []
        pred_vals = []
        for p in range(fmaps.shape[0]):
            pvals = []
            for i in tqdm(range(args.num_train_img)): #input.shape[1]
                probabilities = model.forward(fmaps[p:p+1,i:i+1,:])
                pvals.append(probabilities[0].float().cpu().detach().clone().numpy())

                loss = criterion((p, i), probabilities.cuda(), i, e)
                loss.backward(retain_graph=True)
                tot_loss += loss
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            
        roc = np.mean([roc_auc_score(criterion.bin_mask[p,i].float().cpu().detach().clone().numpy(), pvals[i]) for i in range(args.num_train_img)])
        true_vals.extend(criterion.bin_mask[p,i].float().cpu().detach().clone().numpy().tolist())
        pred_vals.extend(pvals[i].tolist())
        print(f"Average ROC of epoch {e}: {roc}")

        if e % 50 == 0:
            filename = f'{args.experiment_path}/model_checkpoints/model_epoch_{e}.pt'
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tot_loss,
                }, filename)

            scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
            model.blm.add_model(f'epoch_{e}', scores_df)
            model.blm.plot_roc(model_names=[f'epoch_{e}'],params={"save":True,"prefix":f'{args.experiment_path}/accuracy_figs/epoch_{e}_'})
            model.blm.plot(model_names=[f'epoch_{e}'],chart_types=[1,2,3,4,5],params={"save":True,"prefix":f'{args.experiment_path}/accuracy_figs/epoch_{e}_'})
            with open(f'{args.experiment_path}/accuracy_figs/loss_curve_data.pickle', 'wb') as handle:
                pickle.dump(losses, handle)

        optimizer.step()
        print(f"Loss epoch {e}: {tot_loss/args.num_train_img}.")
        losses.append(tot_loss.item()/args.num_train_img)
        epoch_time = time.time() - start_time
        print(f"Epoch {e} running time: {epoch_time}.")

    # Used to load model:
    # checkpoint = torch.load("/home/dnori/cryo_reconstruction/experiments/experiment_400images/model_checkpoints/model_epoch_200.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])

    

    