import projection as proj
import numpy as np
import pickle
import time
from tqdm import tqdm

def time_repeat(f, number=10):
    times = np.zeros((number,))
    res = None
    for _n in tqdm(range(number)):
        ctime = time.time()
        res = f()
        times[_n] = time.time() - ctime
    return times, res

def setup() -> np.ndarray:
    with open('processed_dataset.pickle', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    edm_3d = content['6GL7']['3D_map']
    return edm_3d

def main():
    edm_3d = setup()
    print("Setup completed")

    def generate_pda() -> proj.PDA:
        return proj.point_density_array(edm_3d)
    
    pda_generation_time, pda = time_repeat(generate_pda, number=1)
    print(f"Best PDA generation time: {min(pda_generation_time)}")

    def project_pda():
        return proj.random_projection_pda(pda, batch_size=5)

    # project_pda_time, proj_pda = time_repeat(project_pda, number=10)
    # print(f"Best PDA projection time: {min(project_pda_time)}")
    
    def project_pda_batched():
        return proj.random_projection_batched(pda, batch_size=5)

    project_batched_time, proj_pda_batched = time_repeat(project_pda_batched, number=10)
    print(f"Best batched projection time: {min(project_batched_time)}")

    # if not np.allclose(proj_pda, proj_pda_batched):
    #     print("INCORRECT RESULT")
    #     raise Exception

if __name__=="__main__":
    main()