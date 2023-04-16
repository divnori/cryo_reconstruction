"""
Script to extract data from PDB using data_csvs
"""
import biotite
import biotite.structure.io.pdb as pdb
import glob
import mrcfile
import os
import pandas as pd
import pickle
import shutil
import urllib.request
import requests
from tqdm import tqdm

def download_from_csvs(path):
    
    files = glob.glob(path + "/*.csv")

    data_frame = pd.DataFrame()
    content = []
    for filename in files:
        df = pd.read_csv(filename, index_col=None)
        content.append(df)
    
    data_frame = pd.concat(content)

    for index,row in tqdm(data_frame.iterrows()):
        pdb_id, emdb_id_1, method = row["Entry ID"], row["EMDB Map"], row["Reconstruction Method"]
        emdb_id_2 = emdb_id_1.lower().replace("-","_")
        try:
            os.mkdir(f"data/{pdb_id}")
        except:
            pass
        if method == "SINGLE PARTICLE":
            try:
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_file = f"data/{pdb_id}/{pdb_id}.pdb"
                urllib.request.urlretrieve(pdb_url, pdb_file)

                api_url = f"https://files.rcsb.org/pub/emdb/structures/{emdb_id_1}/map/{emdb_id_2}.map.gz"
                response = urllib.request.urlopen(api_url)
                with open(f"data/{pdb_id}/{pdb_id}.map", "wb") as f:
                    f.write(response.read())

                print(f"Completed download for {pdb_id}.")
            except Exception as e:
                print(f"URL not found for {pdb_id}.")
                print(e)
                shutil.rmtree(f"data/{pdb_id}", ignore_errors=True)

def save_dictionary(path):
    result = {}
    for subdir, dirs, files in os.walk(path):
        if len(files) == 2:
            pdb_id = files[0][:4]
            print(pdb_id)
            metadata = {}
            with mrcfile.open(f"{path}/{pdb_id}/{pdb_id}.map") as mrc:
                map_data = mrc.data
                metadata["3D_map"] = map_data
            pdb_file = pdb.PDBFile.read(f"{path}/{pdb_id}/{pdb_id}.pdb")
            structure = pdb_file.get_structure()[0]
            metadata["structure"] = structure.coord.shape
            result[pdb_id] = metadata

    with open('processed_dataset.pickle', 'wb') as handle:
        pickle.dump(result, handle)

if __name__ == "__main__":
    csv_path = "data_csvs"
    data_path = "data"

    #download_from_csvs(csv_path)
    save_dictionary(data_path)
