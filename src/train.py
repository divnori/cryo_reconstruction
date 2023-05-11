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
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from binary_label_metrics import BinaryLabelMetrics
import model as model
import predictor as predictor
from projection import specific_projection_pda
import so3_utils
import visualization

import time
from tqdm import tqdm

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

        # with open('/home/dnori/cryo_reconstruction/ground_truth/all.pickle', 'rb') as pickle_result:
        #     self.ground_truth = pickle.load(pickle_result)
        
        # if not self.ground_truth:
        #     raise NotImplementedError("Missing ground truth data")
        
        # for i in range(len(self.ground_truth)):
        #     self.ground_truth[i] = torch.from_numpy(self.ground_truth[i])
        
        # self.mses = torch.zeros((fmaps.shape[0], fmaps.shape[1], len(self.ground_truth)))
        # # self.prob = torch.zeros((fmaps.shape[0], fmaps.shape[1], len(self.ground_truth))).cuda()
        
        # for i in range(self.mses.shape[0]):
        #     for j in range(self.mses.shape[1]):
        #         fmap = fmaps[i,j,:,:]
        #         for k in range(self.mses.shape[2]):
        #             gt = self.ground_truth[k]
        #             self.mses[i,j,k] = ((gt - fmap)*(gt-fmap)).mean()
        
        # # print("LOSS")
        # # print(torch.min(self.mses[0,0,:]))
        # # print(torch.min(self.mses[0,1,:]))
        # # print(torch.max(self.mses[0,0,:]))
        # # print(torch.max(self.mses[0,1,:]))


        # # print("MAX_MIN_mse",max(torch.min(self.mses[0,i,:]) for i in range(50)))
        # # print("MIN_MAX_mse",min(torch.max(self.mses[0,i,:]) for i in range(50)))

        # # print(torch.min(self.mses[0,2,:]))
        # self.prob = torch.exp(-.0001*self.mses*self.mses)


        # # print(torch.min(self.prob[0,0,:]))
        # # print(torch.min(self.prob[0,1,:]))
        # # print(torch.max(self.prob[0,0,:]))
        # # print(torch.max(self.prob[0,1,:]))

        # # mm = min([(i, torch.max(self.prob[0,i,:])) for i in range(50)], key=lambda x:x[1])
        # # print("MIN_MAX",mm)

        # # import matplotlib.pyplot as plt
        # # plt.hist(self.prob[0,mm[0],:].cpu().detach().numpy(), 20)
        # # plt.savefig("prob_x2.png")

        # threshold_probs = np.repeat(np.expand_dims(np.expand_dims(np.array([torch.max(self.prob[0,i,:])-0.05 for i in range(fmaps.shape[1])]), axis=1).T, axis=2), self.prob.shape[2], axis=2)

        # self.bin_mask = torch.where(self.prob > torch.from_numpy(threshold_probs), 1, 0)

        # with open("bin_mask.pickle", "wb") as f:
        #     pickle.dump(self.bin_mask, f)

        with open("bin_mask.pickle", "rb") as f:
            self.bin_mask = pickle.load(f)
            self.bin_mask = self.bin_mask.cuda()
            self.bce_weight = self.bin_mask + 9

        # min_1s = min([torch.count_nonzero(self.bin_mask[:,i,:]) for i in range(50)])
        # max_1s = max([torch.count_nonzero(self.bin_mask[:,i,:]) for i in range(50)])
        # print("Min1,", min_1s)
        # print("Max1,", max_1s)
        # exit()
        # print("SUM",torch.sum(self.prob[0,0,:]))

    
    def forward(self, fmap_idx, probabilities):
        p = probabilities[0].float()
        loss = torch.nn.BCELoss()
        return  loss(p, self.bin_mask[fmap_idx[0],fmap_idx[1]].float())
        # loss = 0
        # for i in range(probabilities.shape[1]):
        #     p = probabilities[0,i]
        #     mse = self.mses[fmap_idx[0], fmap_idx[1], i]
        #     loss += p * p * mse
        #     # loss += (p - self.prob[fmap_idx[0], fmap_idx[1], i])**2
        # return loss

class Encoder(nn.Module):

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
        so3_fdim_2 = 16
        s2_kernel_grid = so3_utils.s2_healpix_grid(max_beta=np.inf, rec_level=2)
        so3_kernel_grid = so3_utils.so3_near_identity_grid()

        so3_kernel_grid = so3_utils.so3_near_identity_grid()
        self.layers = nn.Sequential(
            model.S2Convolution(s2_fdim, so3_fdim_1, lmax, s2_kernel_grid),
            e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10),
            nn.BatchNorm1d(165),
            model.SO3Convolution(so3_fdim_1, 1, lmax, so3_kernel_grid),
            # e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10),
            # model.SO3Convolution(so3_fdim_2, 1, lmax, so3_kernel_grid)
        ).cuda()

        self.projector = projector = model.Image2SphereProjector(fmap_shape=(5,125, 125), sphere_fdim=args.sphere_fdim, lmax=args.lmax)

        output_xyx = so3_utils.so3_healpix_grid(rec_level=2).cuda() #rec_level is resolution
        self.sphere_grid = output_xyx
        self.eval_wigners = so3_utils.flat_wigner(lmax, *output_xyx).transpose(0,1).cuda() # for probability calculation

    def forward(self, x):
        # x is 2D orthographic projection
        feat = self.featurizer(x.cuda())
        proj = self.projector(feat[0].cpu())[0,:,:].cuda()
        harmonics = self.layers(proj)
        probabilities = self.compute_probabilities(harmonics) #[1,4608]
        # print(type(probabilities))
        # print(probabilities.shape)
        return probabilities
    
    def compute_probabilities(self, harmonics):
        ''' compute probabilities over grid resulting from SO(3) group convolution'''
        probs = torch.matmul(harmonics, self.eval_wigners).squeeze(1)
        return nn.Sigmoid()(probs)
        # return nn.Softmax(dim=1)(probs)

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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--sphere_fdim', type=int, default=64)
    parser.add_argument('--lmax', type=int, default=4)
    parser.add_argument('--harmonic_coefs', type=int, default=25)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    with open('projections-clean.pickle', 'rb') as pickle_file:
        projection_dict = pickle.load(pickle_file)

    # with open('pdas.pickle', 'rb') as pickle_file:
    #     pda_dict = pickle.load(pickle_file)

    shape = (args.img_shape, args.img_shape)

    
    # input = torch.zeros((len(projection_dict.items()), args.proj_per_img, args.sphere_fdim, args.harmonic_coefs))
    fmaps = torch.zeros((len(projection_dict.items()), args.proj_per_img, args.img_shape, args.img_shape))
    # true_pdas = [] # list of pdas where each pda is (coords, densities)

    for i in range(len(projection_dict.items())):
        pdb_id, projections = list(projection_dict.items())[i]
        # true_pdas.append(pda_dict[pdb_id])
        for j in range(len(projections)):
            fmap = projections[j]
            # proj = projector(torch.from_numpy(fmap[np.newaxis, np.newaxis, : ,:]).float())[0,:,:]
            # visualization.visualize_spherical_projection(fmap, projector, j)
            # input[i, j, :, :] = proj
            fmaps[i, j, :, :] = torch.from_numpy(fmap)

    # input is shape (# of proteins, # of projections per protein, sphere_fdim, # of spherical harmonic coefs)
    # add train test split (perhaps sequence similarity split, start with random)

    # starting with very simple case - training with only one protein
    # input: a projection

    criterion = CustomLoss()
    model = Encoder(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    losses = []

    # for e in range(args.epochs):
    #     optimizer.zero_grad()
    #     start_time = time.time()
    #     model.train()
    #     tot_loss = 0
    #     true_vals = []
    #     pred_vals = []
    #     for p in range(fmaps.shape[0]):
    #         pvals = []
    #         for i in tqdm(range(40)): #input.shape[1]
    #             # print(f"\t\tprojection {i} epoch {e}")
    #             probabilities = model.forward(fmaps[p:p+1,i:i+1,:])
    #             pvals.append(probabilities[0].float().cpu().detach().clone().numpy())
    #             loss = criterion((p, i), probabilities.cuda())
    #             loss.backward(retain_graph=True)
    #             tot_loss += loss
    #             nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    #             # print("sum", torch.sum(criterion.bin_mask[p,i,:]), torch.sum(probabilities[0]))
            
    #     # if e % 20 == 0 or e == args.epochs-1:
    #     roc = np.mean([roc_auc_score(criterion.bin_mask[p,i].float().cpu().detach().clone().numpy(), pvals[i]) for i in range(40)])
    #     true_vals.extend(criterion.bin_mask[p,i].float().cpu().detach().clone().numpy().tolist())
    #     pred_vals.extend(pvals[i].tolist())
    #     print(f"Average ROC of epoch {e}: {roc}")

    #     if roc > 0.77:
    #         rounded_roc = str(round(roc, 2))
    #         filename = f'model_checkpoints/model_epoch_{e}_roc_{rounded_roc}.pt'
    #         torch.save({
    #             'epoch': e,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': tot_loss,
    #             }, filename)

    #         scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
    #         model.blm.add_model(f'epoch_{e}', scores_df)
    #         model.blm.plot_roc(model_names=[f'epoch_{e}'],params={"save":True,"prefix":f"figures/epoch_{e}_{rounded_roc}_"})
    #         model.blm.plot(model_names=[f'epoch_{e}'],chart_types=[1,2,3,4,5],params={"save":True,"prefix":f"figures/epoch_{e}_{rounded_roc}_"})
    #         with open('loss_curve_data.pickle', 'wb') as handle:
    #             pickle.dump(losses, handle)
    #         break

    #     optimizer.step()
    #     # scheduler.step()
    #     print(f"Loss epoch {e}: {tot_loss/40}.")
    #     losses.append(tot_loss.item()/40)
    #     epoch_time = time.time() - start_time
    #     print(f"Epoch {e} running time: {epoch_time}.")

    checkpoint = torch.load("/home/dnori/cryo_reconstruction/model_checkpoints/model_epoch_212_roc_0.77.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # validation
    true_vals = []
    pred_vals = []
    for p in range(fmaps.shape[0]):
        pvals = []
        for i in tqdm(range(40,50)): #input.shape[1]
            probabilities = model.forward(fmaps[p:p+1,i:i+1,:])
            pvals.append(probabilities[0].float().cpu().detach().clone().numpy())

            true_vals.extend(criterion.bin_mask[p,i].float().cpu().detach().clone().numpy().tolist())
            pred_vals.extend(pvals[i-40].tolist())

    
    scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
    model.blm.add_model(f'val', scores_df)
    model.blm.plot_roc(model_names=[f'val'],params={"save":True,"prefix":f"figures/val_"})
    model.blm.plot(model_names=[f'val'],chart_types=[1,2,3,4,5],params={"save":True,"prefix":f"figures/val_"})

    

    