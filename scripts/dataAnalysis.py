import pandas as pd
import numpy as np
import json

#Testing for variations in secondary structure to one amino acid seq

def UniqueAminoAcids(amino_acids):
    unique_aminoacids = []
    for amino_acid in amino_acids:
        if not amino_acid in unique_aminoacids:
            unique_aminoacids.append(amino_acid)
    print('Uniques done')
    return unique_aminoacids
    


def Variations(amino_acids,sec_struct,unique_aminoacids):
    variations = {}
    
    for amino_acid in amino_acids:
        for i,amino_acid_ in enumerate(unique_aminoacids):
            if amino_acid == amino_acid_:
                if not amino_acid in variations:
                    variations[amino_acid] = sec_struct[i]
                else:
                    variations.update({amino_acid:variations[amino_acid] + ',' + sec_struct[i]})
    print('Variations done')
    return variations

#Testing if secondary structure is always same for one amino-acid sequence

def StructureRecurence(variations):
    recurent = {}
    for aminoacid_seq,sec_structures in enumerate(variations):
        recurent[aminoacid_seq] = True
        for i,sec_sequence in enumerate(sec_structures.split(',')):
            if i > 0:
                if sec_sequence != sec_structures[i-1]:
                    recurent.update({aminoacid_seq:False})
                    break
    print('StructureRecurence done')
    return recurent

#Attempt to find common segments of secondary structure for one amino-acid sequence (Finding common paterns)
def CommonSeqments(variations):
    common_seqments = []
    for sec_structs in variations:
        splited_sec_structs = sec_structs.split(',')
        common_seqment = max(splited_sec_structs,key=len)
        for sec_struct in splited_sec_structs:
            for i,char in enumerate(common_seqment):
                if sec_struct[i] != char:
                    common_seqment[i]=='.'
                else:
                    common_seqment[i]==sec_struct[i]
        common_seqments.append(common_seqments)
    print('CommonSeqments done')
    return common_seqments



if __name__=="__main__":
    df = pd.read_csv('data/raw/AMINtoSEC.csv')


    amino_acids = [x[0] for x in np.array(df)]
    sec_struct = [x[1] for x in np.array(df)]

    unique_aminoacids = UniqueAminoAcids(amino_acids)
    variations =  Variations(amino_acids,sec_struct,unique_aminoacids)
    
    array = []
    for sec_structures,acid in enumerate(variations):
        x = 0
        for struct in sec_structures:
            x+=1
        array.append([acid,x])

    x = open('data/processed/variations.json','w')
    x.write(json.dumps(array))
    x.close()
    recurent = json.dumps(StructureRecurence(variations))
    x = open('data/processed/structurerecurence.json','w')
    x.write(recurent)
    x.close()
    common_seqments = json.dumps(CommonSeqments(variations))
    x = open('data/processed/commonseqments.json','w') 
    x.write(common_seqments)
    x.close()
    

    
    