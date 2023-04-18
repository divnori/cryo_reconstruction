"""
1. spherical CNN (S^2 -> R^k)
2. reconstruction network
"""

import torch
import torch.nn as nn
from e3nn import o3
import src.so3_utils as utils

class S2Conv(nn.Module):
  '''S2 group convolution which outputs signal over SO(3) irreps

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass.
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()
    # filter weight parametrized over spatial grid on S2
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_s2_pts]

    # linear projection to convert filter weights to fourier domain
    self.register_buffer(
      "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
    )  # [n_s2_pts, (2*lmax+1)**2]

    # defines group convolution using appropriate irreps
    # note, we set internal_weights to False since we defined our own filter above
    self.lin = o3.Linear(utils.s2_irreps(lmax), utils.so3_irreps(lmax), 
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform S2 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, (2*lmax+1)**2), signal over S2 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
    return self.lin(x, weight=psi)


class SO3Conv(nn.Module):
  '''SO3 group convolution

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas, gammas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()

    # filter weight parametrized over spatial grid on SO3
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_so3_pts]

    # wigner D matrices used to project spatial signal to irreps of SO(3)
    self.register_buffer("D", utils.flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, sum_l^L (2*l+1)**2]

    # defines group convolution using appropriate irreps
    self.lin = o3.Linear(utils.so3_irreps(lmax), utils.so3_irreps(lmax), 
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform SO3 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2), signal over SO3 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
    return self.lin(x, weight=psi)

