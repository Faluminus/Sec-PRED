import os
import logging
import subprocess
import multiprocessing
from Bio import PDB
import gzip

#PATHS

inputFileLIN = '../data/raw/pdb/'
inputFileWIN = inputFileLIN
outputFileWIN = '../data/raw/fulldata.csv'
log_dir = 'logs/'






#COMMANDS
def CommandDSSP(inputFile):
    commandDSSP = f'dssp {inputFile} --output-format dssp'
    return commandDSSP

#LOGGING

# Logger setup
log_file = os.path.join(log_dir, "app.log")
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)





#FUNCTIONS
def CheckForDownloaded():
    with open(os.path.join(log_dir, "app.log"),'r') as logs:
        lineNum = logs.readlines()
        
        
def ProteinPaths(divided_path):
    protein_paths = os.listdir(divided_path)
    return protein_paths


def parse_remark(file_path: str):
    remarks_200 = {}
    with gzip.open(file_path, 'rt', encoding='utf-8') as pdb_file:
        for line in pdb_file:
            if line.startswith("REMARK 200"):
                cut_line = line[len("REMARK 200"):].strip()
                key = ""
                value = ""
                for i,x in enumerate(cut_line):
                    if x == ":":
                        value = cut_line[i+2:]
                        break
                    if x != " ":
                        key += x
                remarks_200[key] = value
    pdb_file.close()
    return remarks_200


def ProcessDSSP(dsspData):
    structure = ""
    aminoacid = ""
    acc = ""
    lines = dsspData.splitlines()
    check = False
    for line in lines:
        if check:
            structure += line[16]
            aminoacid += line[13]
            acc += '|'+line[35:38]
        if line.startswith('  #'):
            check = True
        
    
    return structure, aminoacid, acc


def WorkersRUN(number_of_workers,exec_function):
    download_split = DownloadSplit(number_of_workers)
    pool = multiprocessing.Pool(processes=number_of_workers)
    pool.starmap(exec_function,zip(download_split))


def DownloadSplit(number_of_workers):
    divided_names = os.listdir(inputFileWIN)
    num_entries=0
    for divided in divided_names:
        proteins = ProteinPaths(inputFileWIN + divided)
        for _ in proteins:
            num_entries+=1
    already_processed = len(open(outputFileWIN,'r').readlines()) - 1
    leftovers = num_entries % number_of_workers
    split = (num_entries - leftovers) / number_of_workers

    arr = []
    for worker in range(1,number_of_workers+1):
        if(worker != number_of_workers):
            range_ = [already_processed + worker*split - split, worker*split + leftovers]
            arr.append(range_)
        else:
            range_ = [already_processed + worker*split - split, worker*split]
            arr.append(range_)
    return arr


#MAIN
def main(to_download):
    divided_names = os.listdir(inputFileWIN)
    
    currently_downloading = 0

    print(to_download)
    for divided in divided_names:
        proteins = ProteinPaths(inputFileWIN + divided)
        for protein in proteins:
            if to_download[0] <= currently_downloading and to_download[1] >= currently_downloading:
                logger.info(protein + ':processing')
                remarks = parse_remark(inputFileLIN + '/' + divided + '/' + protein)
                args = subprocess.run(CommandDSSP(inputFileLIN + '/' + divided + '/' + protein),capture_output=True, text=True, shell=True).stdout
                processed_dssp = ProcessDSSP(args)
                if processed_dssp != ['','']:
                    finalDataset = open(outputFileWIN,'a',newline="")
                    finalDataset.write('\n'+processed_dssp[0]+','+processed_dssp[1]+','+(remarks.get('PH') or '')+','+(remarks.get('TEMPERATURE(KELVIN)') or '')+','+processed_dssp[2])
                    finalDataset.close()
            currently_downloading +=1



if __name__ == '__main__': 
    finalDataset = open(outputFileWIN,'a',newline="")
    finalDataset.write("AminoAcidSeq,SecondaryStructureSeq,PH,Kelvin,SolventAcessibility")
    finalDataset.close()
    arr = DownloadSplit(1)
    main(arr[0])
   

