"""
Script to extract data from PDB using data_csvs
"""

import glob
import os
import pandas as pd
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
        pdb_id, emdb_id, method = row["Entry ID"], row["EMDB Map"], row["Reconstruction Method"]
        try:
            os.mkdir(f"data/{pdb_id}")
        except:
            pass
        if method == "SINGLE PARTICLE":
            try:
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_file = f"data/{pdb_id}/{pdb_id}.pdb"
                urllib.request.urlretrieve(pdb_url, pdb_file)

                api_url = f"https://files.rcsb.org/download/{emdb_id}.map"
                response = urllib.request.urlopen(api_url)
                with open(f"data/{pdb_id}/{emdb_id}.map", "wb") as f:
                    f.write(response.read())

                print(f"Completed download for {pdb_id}.")
            except Exception as e:
                print(f"URL not found for {pdb_id, emdb_id}.")
                print(e)
                shutil.rmtree(f"data/{pdb_id}", ignore_errors=True)

def save_dictionary(path):
    files = glob.glob(path + "/*.csv")

if __name__ == "__main__":
    csv_path = "data_csvs"
    data_path = "data"

    download_from_csvs(csv_path)
    #save_dictionary(data_path)
