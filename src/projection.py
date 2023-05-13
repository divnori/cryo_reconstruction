import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import scipy
import torch
import torch.nn as nn
import e3nn
from e3nn import o3
from torchvision.transforms import ToPILImage
import healpy as hp
from math import sin
from math import cos
import pickle
import so3_utils

# point-density representation type
PDR =  list[tuple[np.ndarray, np.float32]]
PDA = tuple[np.ndarray, np.ndarray]
PDA_cuda = tuple[torch.Tensor, torch.Tensor]

def normalize_edm(edm: np.ndarray) -> np.ndarray:
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

def random_projection_pda(
        pda: PDA, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        blur_sigma: float | tuple | None = None,
        noise_sigma: float = 0.1,
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
    
    if blur_sigma is None:
        blur_sigma = (0.005 * shape[0], 0.005 * shape[1])

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
            # if (i%100000 == 0): print(f"pixel {i}/{densities.shape[0]}")
            pixel = coord_to_pixel(rot_coords[i][0], rot_coords[i][1])
            im[pixel] += densities[i]
        
        if noise_sigma > 0:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    noise = np.random.normal(0.0, noise_sigma)
                    im[i,j] += noise * np.abs(im[i, j])
        
        im = scipy.ndimage.gaussian_filter(im, blur_sigma)
        
        images.append(im)
    
    return images


def visualize_projection(projection: np.ndarray):
    import matplotlib.pyplot as plt
    plt.imshow(projection, cmap='hot')
    plt.show()


def random_projection_pda_gt(
        pda: PDA, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        blur_sigma: float | tuple | None = None,
        noise_sigma: float = 0.1,
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

    if blur_sigma is None:
        blur_sigma = (0.005 * shape[0], 0.005 * shape[1])

    images = []
    # print("image is cuda:", images.is_cuda)
    coords, densities = pda
    # densities = densities.numpy()

    for idx, rot in enumerate(rand_rots):
        print(f"Rotation {idx}/{len(rand_rots)}")
        # print(f"Starting rotation {idx}")
        if not isinstance(rot, np.ndarray):
            rot_mtx = rot.as_matrix()
        else:
            rot_mtx = rot
        
        rot_coords = coords @ rot_mtx.T[:,:2]
        # print(rot.as_matrix())
        pixels = (rot_coords + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        pixels = pixels * np.array(shape)
        pixels = np.floor(pixels).astype(int)
        # pixels = pixels.cpu().detach().numpy()

        # print(pixels[0:10,:])
        # pixels = torch.zeros((densities.shape[0], 2), dtype=torch.float32).cuda()
        # for i in range(10):
        #     if (i%10000 == 0): print(f"pixel {i}/{densities.shape[0]}")
        #     pixel = coord_to_pixel(rot_coords[i][0], rot_coords[i][1])
        #     print(pixel)
        #     pixels[i,0] = pixel[0]
        #     pixels[i,1] = pixel[1]
        
        im = np.zeros(shape)
        for i in range(pixels.shape[0]):
            # if (i%100000 == 0): print(f"pixel {i}/{densities.shape[0]}")
            im[pixels[i,0], pixels[i,1]] += densities[i] 

        # if noise_sigma > 0:
        #     for i in range(shape[0]):
        #         for j in range(shape[1]):
        #             noise = np.random.normal(0.0, noise_sigma)
        #             im[i,j] += noise * np.abs(im[i, j])
        
        im = scipy.ndimage.gaussian_filter(im, blur_sigma)

        with open(f'ground_truth_2/image-{idx}.pickle', 'wb') as pickle_result:
            pickle.dump(im, pickle_result)

        images.append(im)
    
    return images

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

    # rotations = np.zeros((poses.shape[1], 3, 3))

    rotations = o3.angles_to_matrix(poses[0], poses[1], poses[2])
    # for p in range(poses.shape[1]):
    #     alpha = poses[0,p]
    #     beta = poses[1,p]
    #     gamma = poses[2,p]

    #     Ry_alpha = np.array([[ cos(alpha),  0,   sin(alpha)],
    #         [ 0,           1,   0         ],
    #         [-sin(alpha),  0,   cos(alpha)]])

    #     Rx = np.array([[1,           0,            0],
    #         [0,  cos(beta),  -sin(beta)],
    #         [0,  sin(beta),   cos(beta)]])

    #     Ry_gamma = np.array([[ cos(gamma),  0,   sin(gamma)],
    #         [ 0,           1,   0         ],
    #         [-sin(gamma),  0,   cos(gamma)]])

    #     R = Ry_gamma @ Rx @ Ry_alpha

    #     rotations[p] = R

    projections = random_projection_pda_gt(pda, shape=(512,512), blur_sigma=(2,2), random=False, rotations=rotations)
    return projections


def random_projection_pda_smart(
        pda: PDA, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        blur_sigma: float | tuple | None = None,
        noise_sigma: float = 0.1,
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

    if blur_sigma is None:
        blur_sigma = (0.005 * shape[0], 0.005 * shape[1])

    images = []
    # print("image is cuda:", images.is_cuda)
    coords, densities = pda
    # densities = densities.numpy()

    for idx, rot in enumerate(rand_rots):
        # print(f"Starting rotation {idx}")
        if not isinstance(rot, np.ndarray):
            rot_mtx = rot.as_matrix()
        else:
            rot_mtx = rot
        
        rot_coords = coords @ rot_mtx.T[:,:2]
        # print(rot.as_matrix())
        pixels = (rot_coords + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        pixels = pixels * np.array(shape)
        pixels = np.floor(pixels).astype(int)
        # pixels = pixels.cpu().detach().numpy()

        # print(pixels[0:10,:])
        # pixels = torch.zeros((densities.shape[0], 2), dtype=torch.float32).cuda()
        # for i in range(10):
        #     if (i%10000 == 0): print(f"pixel {i}/{densities.shape[0]}")
        #     pixel = coord_to_pixel(rot_coords[i][0], rot_coords[i][1])
        #     print(pixel)
        #     pixels[i,0] = pixel[0]
        #     pixels[i,1] = pixel[1]
        
        im = np.zeros(shape)
        for i in range(pixels.shape[0]):
            # if (i%100000 == 0): print(f"pixel {i}/{densities.shape[0]}")
            im[pixels[i,0], pixels[i,1]] += densities[i] 

        # if noise_sigma > 0:
        #     for i in range(shape[0]):
        #         for j in range(shape[1]):
        #             noise = np.random.normal(0.0, noise_sigma)
        #             im[i,j] += noise * np.abs(im[i, j])
        
        im = scipy.ndimage.gaussian_filter(im, blur_sigma)
        images.append(im)
    
    return images


# def random_projection_batched(
#         pda: PDA, 
#         shape: tuple[int, int] = (256, 256),
#         batch_size: int = 1,
#         zoom_scale: float = np.sqrt(3),
#         blur_sigma: float | tuple | None = None,
#         noise_sigma: float = 0.1,
#         seed: int = 69,
#         random: bool = True,
#         rotations=None
#     ) -> list[np.ndarray]:
#     """
#     Takes a PDA as input, and returns a random 2d projection.
#     """
#     if random:
#         rand_rots = Rotation.random(batch_size, seed)
#     else:
#         rand_rots = rotations

#     if blur_sigma is None:
#         blur_sigma = (0.005 * shape[0], 0.005 * shape[1])

#     # takes [-0.5, 0.5]^2 -> resolution
#     def coord_to_pixel(x, y) -> tuple[int, int]:
#         norm_x = (x + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
#         norm_y = (y + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
#         return (int(torch.floor(shape[0] * norm_x)), int(torch.floor(shape[1] * norm_y)))

#     rot_mtx = torch.tensor(rand_rots.as_matrix()).cuda().reshape((batch_size*3, 3)).T

#     # assert rot_mtx.shape == (3, 2*batch_size)

#     images = []
#     coords, densities = pda
#     rot_coords = coords @ rot_mtx

#     # norm_coords = (rot_coords + 0.5)/ zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
#     # shape_arr = np.array(shape)

#     for b in range(batch_size):

#         im = np.zeros(shape)
#         for i in range(densities.shape[0]):
#             pixel = coord_to_pixel(rot_coords[i][3*b], rot_coords[i][3*b+1])
#             im[pixel] += densities[i]
        
#         if noise_sigma > 0:
#             for i in range(shape[0]):
#                 for j in range(shape[1]):
#                     noise = np.random.normal(0.0, noise_sigma)
#                     im[i,j] += noise * np.abs(im[i, j])
        
#         im = scipy.ndimage.gaussian_filter(im, blur_sigma)
#         images.append(im)
    
#     return images
    
def random_projection_pda_cuda(
        pda: PDA_cuda, 
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        zoom_scale: float = np.sqrt(3),
        blur_sigma: float | tuple | None = None,
        noise_sigma: float = 0.1,
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

    if blur_sigma is None:
        blur_sigma = (0.005 * shape[0], 0.005 * shape[1])

    images = []
    coords, densities = pda

    for rot in rand_rots:
        # print(f"Starting rotation {idx}")
        if not isinstance(rot, torch.Tensor):
            rot_mtx = torch.tensor(rot.as_matrix()).cuda()
        else:
            rot_mtx = rot
        
        rot_coords = coords @ rot_mtx.T[:,:2]
        # print(rot.as_matrix())
        pixels = (rot_coords + 0.5) / zoom_scale + (zoom_scale - 1) / (2 * zoom_scale)
        pixels = pixels * torch.tensor(shape).cuda()
        pixels = torch.floor(pixels).to(dtype=torch.int32)
        pixels = pixels.cpu().detach().numpy()

        # print(pixels[0:10,:])
        # pixels = torch.zeros((densities.shape[0], 2), dtype=torch.float32).cuda()
        # for i in range(10):
        #     if (i%10000 == 0): print(f"pixel {i}/{densities.shape[0]}")
        #     pixel = coord_to_pixel(rot_coords[i][0], rot_coords[i][1])
        #     print(pixel)
        #     pixels[i,0] = pixel[0]
        #     pixels[i,1] = pixel[1]
        
        im = np.zeros(shape)
        for i in range(pixels.shape[0]):
            # if (i%100000 == 0): print(f"pixel {i}/{densities.shape[0]}")
            im[pixels[i,0], pixels[i,1]] += densities[i] 

        # if noise_sigma > 0:
        #     for i in range(shape[0]):
        #         for j in range(shape[1]):
        #             noise = np.random.normal(0.0, noise_sigma)
        #             im[i,j] += noise * np.abs(im[i, j])
        
        im = scipy.ndimage.gaussian_filter(im, blur_sigma)
        images.append(im)
    
    return images


def generate_ground_truth():
    with open('pdas.pickle', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    pda = content['6bdf']
    sphere_grid = so3_utils.so3_healpix_grid(rec_level=2)
    result = specific_projection_pda(sphere_grid, pda)

    with open('ground_truth_2/all.pickle', 'wb') as pickle_result:
        pickle.dump(result, pickle_result)


if __name__=="__main__":
    # import time
    # ctime = time.time()
    # generate_ground_truth()
    # etime = time.time() - ctime
    # print(f"{etime=}")
    
    # import matplotlib.pyplot as plt
    # for i in range(3000,3005):
    #     with open(f'ground_truth/image-{i}.pickle', 'rb') as pickle_result:
    #         content = pickle.load(pickle_result)
    #     # print(content)
    #     plt.imshow(content, cmap='hot')
    #     plt.savefig(f'example-{i}.png')

    with open('/home/dnori/cryo_reconstruction/ground_truth/all.pickle', 'rb') as pickle_result:
        ground_truth = pickle.load(pickle_result)

    with open('/home/dnori/cryo_reconstruction/projections.pickle', 'rb') as pickle_result:
        projections = pickle.load(pickle_result)

    for k,v in projections.items():
        i = 0
        for proj in v:
            i+=1
            all_mses = []
            for gt in ground_truth:
                all_mses.append(((proj-gt)**2).mean())
            all_mses_arr = np.array(all_mses)
            min_indices = np.argpartition(all_mses_arr, 5)[:5]
            print(all_mses_arr[min_indices])
            for indx in list(min_indices):
                plt.imshow(ground_truth[indx], cmap='hot')
                plt.title(all_mses[indx])
                plt.savefig(f"/home/dnori/cryo_reconstruction/min_max_imgs/min_{indx}_{i}")
            plt.imshow(proj, cmap='hot')
            plt.savefig(f"/home/dnori/cryo_reconstruction/min_max_imgs/proj_{i}")
            