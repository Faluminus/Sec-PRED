import requests
from urllib.request import urlretrieve
import os 
import logging
import pandas as pd
import time

ids = pd.read_csv("data/raw/ids.csv")
print(ids)
ac_with_phylogenics = "data/raw/ac_phylo.txt"
log_dir = "scripts\logs"


def FTPDownload(id):
    ftp = f"https://www.rcsb.org/fasta/entry/{id}/download"
    data = str(requests.get(ftp).content)
    print(data)
    return data

# Logger setup
log_file = os.path.join(log_dir, "ac_phylo.log")
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


while True:
    last_downloaded = 0
    with open(os.path.join(log_dir, "ac_phylo.log")) as f:
        last_downloaded = sum(1 for _ in f)
    
    with open(ac_with_phylogenics,'a') as file:
        try:
            for id_ in ids['identifiers'][last_downloaded:]:
                ftp = FTPDownload(id_)
                file.write(ftp + '\r')
                logger.info('done')
        except:
            time.sleep(5000)
        
