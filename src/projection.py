import numpy as np
from scipy.spatial.transform import Rotation
import scipy
from e3nn import o3
import pickle
import torch
# import so3_utils

# point-density array type
PDA = tuple[np.ndarray, np.ndarray]

def normalize_edm(edm: np.ndarray) -> np.ndarray:
    """
    Normalize the electron density map to separate non-zero values from zero values.
    """
    norm_min = np.min(edm[edm != 0])
    norm_mdn = np.median(edm[edm != 0])
    normalize_item = lambda x: (x - norm_min) / (norm_mdn - norm_min) if x != 0 else x 
    vec_normalize_item = np.vectorize(normalize_item)
    edm = vec_normalize_item(edm)
    return edm

def point_density_array(edm: np.ndarray) -> PDA:
    """
    Converts a 3-dimensional electron density map into a point-density array.
    This representation consists of a two parallel arrays, one with coordinates 
    (normalized to [-0.5, 0.5]^3), and another with density values.
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


def project_pda_to_image(
        pda: PDA, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        blur_sigma: float | tuple | None = None,
        noise_sigma: float = 0.1,
        seed: int = 0,
        random: bool = True,
        rotations=None
    ) -> list[np.ndarray]:
    """
    Takes a PDA as input, and returns 2D projections, which can be randomized or given as input.

    ## Parameters
    shape: tuple of ints
        The shape of each output image
    batch_size: int
        The number of random projections to generate
    zoom_scale: float
        How much to zoom out of the image after projection
    blur_sigma: float or tuple of float
        The standard deviation of Gaussian blur effects
    noise_sigma: float
        The (relative) standard deviation of pointwise Gaussian noise effects
    seed: int
        The seed to use for randomization, to produce reproducable results
    random: bool
        Whether to use random projections or the given rotations
    rotations: array of shape (*, 3, 3) | None
        The rotations used to generate projections if random=True

    ## Returns
    images: list[np.ndarray]
        List of projected images of specified length, each with specified shape
    """
    if random:
        rand_rots = Rotation.random(batch_size, seed)
    else:
        rand_rots = rotations

    if blur_sigma is None:
        blur_sigma = (0.005 * shape[0], 0.005 * shape[1])

    images = []
    coords, densities = pda

    for idx, rot in enumerate(rand_rots):
        print(f"Starting rotation {idx}")
        if not isinstance(rot, np.ndarray):
            rot_mtx = rot.as_matrix()
        else:
            rot_mtx = rot
        
        rot_coords = coords @ rot_mtx.T[:,:2]
        pixels = (rot_coords + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        pixels = pixels * np.array(shape)
        pixels = np.floor(pixels).astype(int)
        
        im = np.zeros(shape)
        for i in range(pixels.shape[0]):
            im[pixels[i,0], pixels[i,1]] += densities[i] 

        if noise_sigma > 0:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    noise = np.random.normal(0.0, noise_sigma)
                    im[i,j] += noise * np.abs(im[i, j])
        
        im = scipy.ndimage.gaussian_filter(im, blur_sigma)
        images.append(im)
    
    return images


def project_pda_to_image_gt(
        pda: PDA, 
        gt_rotations: np.ndarray,
        shape: tuple[int, int] = (256, 256),
        zoom_scale: float = np.sqrt(3),
        blur_sigma: float | tuple | None = None,
    ) -> list[np.ndarray]:
    """
    Takes a PDA as input, and returns ground truth 2D projections. 
    Modified from `project_pda_to_image` to include saving to files and printing progess.

    ## See also 
    project_pda_to_image
    """

    if blur_sigma is None:
        blur_sigma = (0.005 * shape[0], 0.005 * shape[1])

    images = []
    coords, densities = pda

    for idx, rot_mtx in enumerate(gt_rotations):
        print(f"Rotation {idx}/{len(gt_rotations)}")
        
        rot_coords = coords @ rot_mtx.T[:,:2]
        pixels = (rot_coords + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        pixels = pixels * np.array(shape)
        pixels = np.floor(pixels).astype(int)
        
        im = np.zeros(shape)
        for i in range(pixels.shape[0]):
            im[pixels[i,0], pixels[i,1]] += densities[i] 
        
        im = scipy.ndimage.gaussian_filter(im, blur_sigma)

        with open(f'ground_truth/image-{idx}.pickle', 'wb') as pickle_result:
            pickle.dump(im, pickle_result)

        images.append(im)
    
    return images

def generate_ground_truth(rec_level=2):
    """
    Generate ground truth poses and respective projections.
    """
    with open('../data/pda.pickle', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    pda = content['6bdf']

    sphere_grid = so3_utils.so3_healpix_grid(rec_level=rec_level)
    rotation_matrices = o3.angles_to_matrix(sphere_grid[0], sphere_grid[1], sphere_grid[2])
    result = project_pda_to_image_gt(pda, rotation_matrices.numpy(), shape=(512,512), blur_sigma=(2,2))

    with open('../ground_truth/all.pickle', 'wb') as pickle_result:
        pickle.dump(result, pickle_result)


if __name__=="__main__":
    import time
    ctime = time.time()
    generate_ground_truth()
    etime = time.time() - ctime
    print(f"{etime=}")
    
    # import matplotlib.pyplot as plt
    # for i in range(3000,3005):
    #     with open(f'ground_truth/image-{i}.pickle', 'rb') as pickle_result:
    #         content = pickle.load(pickle_result)
    #     # print(content)
    #     plt.imshow(content, cmap='hot')
    #     plt.savefig(f'ground_truth/example-{i}.png')

    # with open('/home/dnori/cryo_reconstruction/ground_truth/all.pickle', 'rb') as pickle_result:
    #     ground_truth = pickle.load(pickle_result)

    # with open('/home/dnori/cryo_reconstruction/projections.pickle', 'rb') as pickle_result:
    #     projections = pickle.load(pickle_result)

    # for k,v in projections.items():
    #     i = 0
    #     for proj in v:
    #         i+=1
    #         all_mses = []
    #         for gt in ground_truth:
    #             all_mses.append(((proj-gt)**2).mean())
    #         all_mses_arr = np.array(all_mses)
    #         min_indices = np.argpartition(all_mses_arr, 5)[:5]
    #         print(all_mses_arr[min_indices])
    #         for indx in list(min_indices):
    #             plt.imshow(ground_truth[indx], cmap='hot')
    #             plt.title(all_mses[indx])
    #             plt.savefig(f"/home/dnori/cryo_reconstruction/min_max_imgs/min_{indx}_{i}")
    #         plt.imshow(proj, cmap='hot')
    #         plt.savefig(f"/home/dnori/cryo_reconstruction/min_max_imgs/proj_{i}")
            