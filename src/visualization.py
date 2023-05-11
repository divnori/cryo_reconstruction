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

def visualize_projection(fmap, projector, i):
    fig = plt.figure(figsize=(10,3))
    
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 101)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    harmonics = projector(torch.from_numpy(fmap[np.newaxis, np.newaxis, : ,:]).float())
    to_grid = o3.ToS2Grid(projector.lmax, (100, 101))
    signal = to_grid(harmonics.squeeze(0).detach()).permute(2, 1, 0).numpy()
    signal = (signal - signal.min())/(signal.max() - signal.min())

    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3.plot_surface(x, y, z, facecolors=signal)

    ax3.set_title('harmonics')

    

def visualize_spherical_projection(fmap, projector, i):
    fig = plt.figure(figsize=(10,3))
    #   ax1 = fig.add_subplot(1,3,1)
    #   ax1.imshow(ToPILImage()(fmap))
    #   ax1.set_title('fmap')

    #   ax2 = fig.add_subplot(1,3,2)
    #   ax2.scatter(*projector.xyz.T[[0,2]])
    #   ax2.set_xlim(-1.1, 1.1)
    #   ax2.set_ylim(-1.1, 1.1)
    #   ax2.set_aspect('equal')
    #   ax2.set_title('grid')

    #   # plot signal on sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 101)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    harmonics = projector(torch.from_numpy(fmap[np.newaxis, np.newaxis, : ,:]).float())
    to_grid = o3.ToS2Grid(projector.lmax, (100, 101))
    signal = to_grid(harmonics.squeeze(0).detach()).permute(2, 1, 0).numpy()
    signal = (signal - signal.min())/(signal.max() - signal.min())
    zeros_layer = np.zeros((101, 100, 1))
    signal_3channel = np.concatenate((zeros_layer, signal, zeros_layer), axis=2)

    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3.plot_surface(x, y, z, facecolors=signal_3channel)
    
    ax3.set_title(f"harmonics")

    fig.savefig(f'/home/dnori/cryo_reconstruction/sphere_projections/proj_{i}.png')

def loss_curve(loss_pickle):

    with open('/home/dnori/cryo_reconstruction/loss_curve_data.pickle', 'rb') as pickle_result:
        loss_list = pickle.load(pickle_result)

    data = list(zip([e for e in range(len(loss_list))], loss_list))
    df = pd.DataFrame(data, columns=['epoch','train loss'])
    sns_plot = sns.lineplot(data=df, x='epoch', y='train loss')
    sns_plot.figure.savefig("figures/loss_curve.png")

if __name__ == "__main__":
    loss_curve("/home/dnori/cryo_reconstruction/loss_curve_data.pickle")
