import numpy as np
from scipy.spatial.transform import Rotation

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
        resolution: tuple[int, int]=(256, 256),
        distance_weighting=False
    ) -> np.ndarray:
    """
    Takes a 3-dimensional electron density map as input, and returns a random 2d 
    projection.
    """
    rand_rot = Rotation.random()
    
    im = np.zeros(resolution)

    # TODO: allow for more padding
    # takes [-0.5, 0.5]^2 -> resolution
    def coord_to_pixel(x, y) -> tuple[int, int]:
        return (int(np.floor(resolution[0] * (x + 0.5))), int(np.floor(resolution[1] * (y + 0.5))))

    for loc, den in pdr:
        rot_loc = rand_rot.apply(loc)
        pixel = coord_to_pixel(rot_loc[0], rot_loc[1])
        if distance_weighting: pass # TODO: implement this
        im[pixel] += den
    
    return im

