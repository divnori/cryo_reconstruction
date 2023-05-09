import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import e3nn
from e3nn import o3
from torchvision.transforms import ToPILImage
import healpy as hp
from math import sin
from math import cos

# point-density representation type
PDR =  list[tuple[np.ndarray, np.float32]]
PDA = tuple[np.ndarray, np.ndarray]

def normalize_edm(edm: np.ndarray) -> np.ndarray:
    norm_min = np.min(edm[edm != 0])
    norm_mdn = np.median(edm[edm != 0])
    normalize_item = lambda x: (x - norm_min) / (norm_mdn - norm_min) if x != 0 else x 
    vec_normalize_item = np.vectorize(normalize_item)
    edm = vec_normalize_item(edm)
    return edm

def point_density_representation(edm: np.ndarray) -> PDR:
    """
    Converts a 3-dimensional electron density map into a point-density representation.
    This representation consists of a dictionary mapping coordinates (normalized to 
    [-0.5, 0.5]^3) to density values.
    """
    assert edm.ndim == 3, "edm must have rank 3"

    nonzero_coords = np.nonzero(edm)
    pdr = [
       ((np.array([i / edm.shape[0] - 0.5, j / edm.shape[1] - 0.5, k / edm.shape[2] - 0.5]), edm[i,j,k])) 
       for i,j,k in zip(*nonzero_coords)
    ]
    
    return pdr
    
def point_density_array(edm: np.ndarray) -> PDA:
    """
    Converts a 3-dimensional electron density map into a point-density array.
    This representation consists of a two parallel arrays, one with coordinates 
    (normalized to [-0.5, 0.5]^3), and another wit density values.
    """
    assert edm.ndim == 3, "edm must have rank 3"

    nonzero_coords = np.nonzero(edm)

    coords = np.array([
       [i / edm.shape[0] - 0.5, j / edm.shape[1] - 0.5, k / edm.shape[2] - 0.5]
       for i,j,k in zip(*nonzero_coords)
    ])
    densities = np.array([
        edm[i,j,k] for i,j,k in zip(*nonzero_coords)
    ])
    
    return (coords, densities)

def random_projection_pdr(
        pdr: PDR, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        distance_weighting = False,
        noise_stddev: float = 0,
        seed: int = 69,
    ) -> list[np.ndarray]:
    """
    Takes a PDR as input, and returns a random 2d projection.
    """
    
    rand_rots = Rotation.random(batch_size, seed)

    # takes [-0.5, 0.5]^2 -> resolution
    def coord_to_pixel(x, y) -> tuple[int, int]:
        norm_x = (x + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        norm_y = (y + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        return (int(np.floor(shape[0] * norm_x)), int(np.floor(shape[1] * norm_y)))

    images = []

    for rot in rand_rots:
        im = np.zeros(shape)
        # print(rot.as_matrix())
        for loc, den in pdr:
            rot_loc = rot.apply(loc)
            pixel = coord_to_pixel(rot_loc[0], rot_loc[1])
            if distance_weighting: pass # TODO: implement this
            im[pixel] += den
        # TODO: add noise
        if noise_stddev > 0:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    noise = np.random.normal(0.0, noise_stddev)
                    im[i,j] += noise * np.abs(im[i, j])
        images.append(im)
    
    return images

def random_projection_pda(
        pda: PDA, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        distance_weighting = False,
        noise_stddev: float = 0,
        seed: int = 69,
        random: bool = True,
        rotations=None
    ) -> list[np.ndarray]:
    """
    Takes a PDA as input, and returns a random 2d projection.
    """
    
    if random:
        rand_rots = Rotation.random(batch_size, seed)
    else:
        rand_rots = rotations

    # takes [-0.5, 0.5]^2 -> resolution
    def coord_to_pixel(x, y) -> tuple[int, int]:
        norm_x = (x + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        norm_y = (y + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        return (int(np.floor(shape[0] * norm_x)), int(np.floor(shape[1] * norm_y)))

    images = []
    coords, densities = pda

    for rot in rand_rots:
        if not isinstance(rot, np.ndarray):
            rot_mtx = rot.as_matrix()
        else:
            rot_mtx = rot
                    
        rot_coords = coords @ rot_mtx.T[:,:2]
        im = np.zeros(shape)
        # print(rot.as_matrix())
        for i in range(densities.shape[0]):
            pixel = coord_to_pixel(rot_coords[i][0], rot_coords[i][1])
            if distance_weighting: pass # TODO: implement this
            im[pixel] += densities[i]
        
        if noise_stddev > 0:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    noise = np.random.normal(0.0, noise_stddev)
                    im[i,j] += noise * np.abs(im[i, j])
        images.append(im)
    
    return images

def random_projection_batched(
        pda: PDA, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        distance_weighting = False,
        noise_stddev: float = 0,
        seed: int = 69
    ) -> list[np.ndarray]:
    """
    Takes a PDA as input, and returns a random 2d projection.
    """
    rand_rots = Rotation.random(batch_size, seed)

    # takes [-0.5, 0.5]^2 -> resolution
    def coord_to_pixel(x, y) -> tuple[int, int]:
        norm_x = (x + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        norm_y = (y + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        return (int(np.floor(shape[0] * norm_x)), int(np.floor(shape[1] * norm_y)))

    rot_mtx = rand_rots.as_matrix()[:,0:2,:].reshape((batch_size*2, 3)).T

    assert rot_mtx.shape == (3, 2*batch_size)

    images = []
    coords, densities = pda
    coords.cuda()   
    rot_coords = coords @ rot_mtx

    for b in range(batch_size):
        im = np.zeros(shape)
        for i in range(densities.shape[0]):
            pixel = coord_to_pixel(rot_coords[i][2*b], rot_coords[i][2*b+1])
            if distance_weighting: pass # TODO: implement this
            im[pixel] += densities[i]
        
        if noise_stddev > 0:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    noise = np.random.normal(0.0, noise_stddev)
                    im[i,j] += noise * np.abs(im[i, j])
        
        images.append(im)
    
    return images

def visualize_projection(projection: np.ndarray):
    import matplotlib.pyplot as plt
    plt.imshow(projection, cmap='hot')
    plt.show()

def specific_projection_pda(
        poses: torch.Tensor,
        pda: np.ndarray
    ) -> list[np.ndarray]:
    """
    poses: a tensor of shape (3,N) where each column represents a pose on sphere
    specified by [alpha, beta, gamma]
    return list of projections of PDA from these poses
    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y
    """

    rotations = np.zeros((poses.shape[1], 3, 3))
    for p in range(poses.shape[1]):
        alpha = poses[0,p]
        beta = poses[1,p]
        gamma = poses[2,p]

        Rx = np.array([[1,           0,            0],
            [0,  cos(beta),  -sin(beta)],
            [0,  sin(beta),   cos(beta)]])

        Ry = np.array([[ cos(alpha),  0,   sin(alpha)],
            [ 0,           1,   0         ],
            [-sin(alpha),  0,   cos(alpha)]])

        Rz = np.array([[cos(gamma),  -sin(gamma),  0],
            [sin(gamma),   cos(gamma),  0],
            [0,            0,           1]])

        R = Rz @ Ry @ Rx

        rotations[p] = R

    projections = random_projection_pda(pda, shape=(512,512), random=True, rotations=rotations)
    return projections

    
