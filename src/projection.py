import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import e3nn
from e3nn import o3
import healpy as hp

# point-density representation type
PDR =  list[tuple[np.ndarray, np.float32]]

def point_density_representation(edm: np.ndarray) -> PDR:
    """
    Converts a 3-dimensional electron density map into a point-density representation.
    This representation consists of a dictionary mapping coordinates (normalized to 
    [-0.5, 0.5]^3) to density values.
    """
    assert edm.ndim == 3, "edm must have rank 3"

    pdr = []
    for i in range(edm.shape[0]):
        for j in range(edm.shape[1]):
            for k in range(edm.shape[2]):
                coord_norm = np.array([i / edm.shape[0] - 0.5, j / edm.shape[1] - 0.5, k / edm.shape[2] - 0.5])
                if edm[i,j,k] != 0:
                    pdr.append((coord_norm, edm[i,j,k]))

    return pdr
    

def random_projection(
        pdr: PDR, 
        shape: tuple[int, int] = (256, 256),
        distance_weighting = False,
        noise_stddev: float = 0
    ) -> np.ndarray:
    """
    Takes a 3-dimensional electron density map as input, and returns a random 2d 
    projection.
    """
    rand_rot = Rotation.random()
    
    im = np.zeros(shape)

    # TODO: allow for more padding
    # takes [-0.5, 0.5]^2 -> resolution
    def coord_to_pixel(x, y) -> tuple[int, int]:
        return (int(np.floor(shape[0] * (x + 0.5))), int(np.floor(shape[1] * (y + 0.5))))

    for loc, den in pdr:
        rot_loc = rand_rot.apply(loc)
        pixel = coord_to_pixel(rot_loc[0], rot_loc[1])
        if distance_weighting: pass # TODO: implement this
        im[pixel] += den

    # TODO: add blurring
    for i in range(np.shape[0]):
       for j in range(np.shape[1]):
          noise = np.random.normal(0.0, noise_stddev)
          

    
    return im

def s2_healpix_grid(rec_level: int=0, max_beta: float=np.pi/6):
    """Returns healpix grid up to a max_beta
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    m = hp.query_disc(nside=n_side, vec=(0,0,1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()


class Image2SphereProjector(nn.Module):
  '''Define orthographic projection from image space to half of sphere, returning
  coefficients of spherical harmonics

  :fmap_shape: shape of incoming feature map (channels, height, width)
  :fdim_sphere: dimensionality of featuremap projected to sphere
  :lmax: maximum degree of harmonics
  :coverage: fraction of feature map that is projected onto sphere
  :sigma: stdev of gaussians used to sample points in image space
  :max_beta: maximum azimuth angle projected onto sphere (np.pi/2 corresponds to half sphere)
  :taper_beta: if less than max_beta, taper magnitude of projected features beyond this angle
  :rec_level: recursion level of healpy grid where points are projected
  :n_subset: number of grid points used to perform projection, acts like dropout regularizer
  '''
  def __init__(self,
               fmap_shape, 
               sphere_fdim: int,
               lmax: int,
               coverage: float = 0.9,
               sigma: float = 0.2,
               max_beta: float = np.radians(90),
               taper_beta: float = np.radians(75),
               rec_level: int = 2,
               n_subset: int = 20,
              ):
    super().__init__()
    self.lmax = lmax
    self.n_subset = n_subset

    # point-wise linear operation to convert to proper dimensionality if needed
    if fmap_shape[0] != sphere_fdim:
      self.conv1x1 = nn.Conv2d(fmap_shape[0], sphere_fdim, 1)
    else:
      self.conv1x1 = nn.Identity()

    # determine sampling locations for orthographic projection
    self.kernel_grid = s2_healpix_grid(max_beta=max_beta, rec_level=rec_level)
    self.xyz = o3.angles_to_xyz(*self.kernel_grid)

    # orthographic projection
    max_radius = torch.linalg.norm(self.xyz[:,[0,2]], dim=1).max()
    sample_x = coverage * self.xyz[:,2] / max_radius # range -1 to 1
    sample_y = coverage * self.xyz[:,0] / max_radius

    gridx, gridy = torch.meshgrid(2*[torch.linspace(-1, 1, fmap_shape[1])], indexing='ij')
    scale = 1 / np.sqrt(2 * np.pi * sigma**2)
    data = scale * torch.exp(-((gridx.unsqueeze(-1) - sample_x).pow(2) \
                                +(gridy.unsqueeze(-1) - sample_y).pow(2)) / (2*sigma**2) )
    data = data / data.sum((0,1), keepdims=True)

    # apply mask to taper magnitude near border if desired
    betas = self.kernel_grid[1]
    if taper_beta < max_beta:
        mask = ((betas - max_beta)/(taper_beta - max_beta)).clamp(max=1).view(1, 1, -1)
    else:
        mask = torch.ones_like(data)

    data = (mask * data).unsqueeze(0).unsqueeze(0).to(torch.float32)
    self.weight = nn.Parameter(data= data, requires_grad=True)

    self.n_pts = self.weight.shape[-1]
    self.ind = torch.arange(self.n_pts)

    self.register_buffer(
        "Y", o3.spherical_harmonics_alpha_beta(range(lmax+1), *self.kernel_grid, normalization='component')
    )

  def forward(self, x):
    '''
    :x: float tensor of shape (B, C, H, W)
    :return: feature vector of shape (B,P,C) where P is number of points on S2
    '''
    x = self.conv1x1(x)

    if self.n_subset is not None:
        self.ind = torch.randperm(self.n_pts)[:self.n_subset]

    x = (x.unsqueeze(-1) * self.weight[..., self.ind]).sum((2,3))
    x = torch.relu(x)
    x = torch.einsum('ni,xyn->xyi', self.Y[self.ind], x) / self.ind.shape[0]**0.5
    return x