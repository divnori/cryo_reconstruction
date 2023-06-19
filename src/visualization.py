"""
Visualization functions
"""
from e3nn import o3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import torch
from torchvision.transforms import ToPILImage
import projection as proj
import train
from scipy.spatial.transform import Rotation
import so3_utils

def loss_curve(loss_pickle):
    """
    Visualize loss curve using pickle of loss lists through epochs
    """
    with open(loss_pickle, 'rb') as pickle_result:
        loss_list = pickle.load(pickle_result)

    data = list(zip([e for e in range(len(loss_list))], loss_list))
    df = pd.DataFrame(data, columns=['epoch','train loss'])
    sns_plot = sns.lineplot(data=df, x='epoch', y='train loss')
    sns_plot.figure.savefig("figures/loss_curve.png")

def test_equivariance(pda, model, randomized=False):
    """
    Tests if probability distributions are equivariant with respect to projection images
    """
    seed = 1
    rot1 = Rotation.random(1, seed).as_matrix()
    p1 = proj.project_pda_to_image(pda, shape=(512, 512), noise_sigma=0, random=False, rotations=rot1)[0]

    a, b, g = torch.tensor([np.pi/6]), torch.tensor([np.pi/6]), torch.tensor([np.pi/6])
    mat = o3.angles_to_matrix(a, b, g)[0]
    rot2 = Rotation.random(1, seed).as_matrix() if randomized else mat.numpy() @ rot1
    p2 = proj.project_pda_to_image(pda, shape=(512, 512), noise_sigma=0, random=False, rotations=rot2)[0]

    p1 = torch.tensor(p1[np.newaxis, np.newaxis, :, :]).float()
    p2 = torch.tensor(p2[np.newaxis, np.newaxis, :, :]).float()

    d1 = model.forward(p1)[0].cpu().detach().numpy()
    d2 = model.forward(p2)[0].cpu().detach().numpy()

    so3_grid = so3_utils.so3_healpix_grid(rec_level=2)
    mat_grid = o3.angles_to_matrix(so3_grid[0], so3_grid[1], so3_grid[2])
    mat_grid_2 = mat @ mat_grid

    map_2_to_1 = [so3_utils.nearest_rotmat(mat_grid_2[i:i+1], mat_grid) for i in range(mat_grid_2.shape[0])]
    
    scores = [(d2[i] - d1[map_2_to_1[i]])**2 for i in range(d2.shape[0])]
    return np.mean(scores)

if __name__ == "__main__":    
    pass

    

    
