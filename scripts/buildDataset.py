import os
import logging
import subprocess
import multiprocessing

#PATHS

inputFileLIN = '../data/raw/mmCIF'
inputFileWIN = inputFileLIN
outputFileWIN = 'data/raw/AMINtoSEC.csv'
log_dir = 'scripts/logs/'






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

def ProcessDSSP(dsspData):
    allData = ["",""]
    lines = dsspData.splitlines()
    i = False
    for line in lines:

        if i:
            structure = line[16]
            aminoacid = line[13]
            allData[0] += aminoacid
            allData[1] += structure

        if line.startswith('  #'):
            i = True
        
    
    return allData


def WorkersRUN(number_of_workers,exec_function):
    download_split = DownloadSplit(number_of_workers)
    pool = multiprocessing.Pool(processes=number_of_workers)
    pool.starmap(exec_function,zip(download_split))



def DownloadSplit(number_of_workers):
    divided_names = os.listdir(inputFileWIN)
    x=0
    for divided in divided_names:
        proteins = ProteinPaths(inputFileWIN + divided)
        for protein in proteins:
            x+=1
    leftovers = x % number_of_workers
    
    split = (x - leftovers) / number_of_workers

    arr = []
    for worker in range(1,number_of_workers+1):
        if(worker != number_of_workers):
            range_ = [worker*split - split ,worker*split + leftovers]
            arr.append(range_)
        else:
            range_ = [worker*split - split ,worker*split]
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
                args = subprocess.run(CommandDSSP(inputFileLIN + '/' + divided + '/' + protein),capture_output=True, text=True, shell=True).stdout
                processed_dssp = ProcessDSSP(args)
                if processed_dssp != ['','']:
                    finalDataset = open(outputFileWIN,'a',newline="")
                    finalDataset.write('\n'+processed_dssp[0]+','+processed_dssp[1])
                    finalDataset.close()
            currently_downloading +=1

if __name__ == '__main__':
    finalDataset = open(outputFileWIN,'a',newline="")
    finalDataset.write("AminoAcidSeq,SecondaryStructureSeq")
    finalDataset.close()

    WorkersRUN(12,main)