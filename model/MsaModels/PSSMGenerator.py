import os
import subprocess

class RunBlast():
    __cmd = "psiblast -query ./swissprot/file.fasta -db swissprot/swissprot -num_iterations 3 -out_ascii_pssm ./swissprot/pssm.txt -evalue 1e-5"
    __path = "./swissprot/pssm.txt"
    def __call__(self):
        os.system(self.__cmd)
        with open(self.__path, "r") as pssm:
            uncleaned_pssm = pssm.read()
            cleaned_pssm = self.__parse_output(uncleaned_pssm)
            return cleaned_pssm
        
    def __parse_output(self, uncleaned_pssm):
        print(uncleaned_pssm)
        


RunBlast()()