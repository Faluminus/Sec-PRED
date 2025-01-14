import pandas as pd
from functools import map, reduce

blosum_pth = "./msa/blosum62.csv"
BLOSUM = pd.read_csv(blosum_pth)

class PhylogenicTree():
    def __init__(self, seq_a= str | None, seq_b= str | None, parent=PhylogenicTree | None):
        self.seq_a = seq_a
        self.seq_b = seq_b
        self.parent = parent
         

class FileMod():
    def __init__(amino_acids_pth):
        all_amino_acids = pd.read_csv(amino_acids_pth)["AminoAcidSeq"]
        print(all_amino_acids)

    def sort_dataset(all_amino_acids):
        pass
        

class msa():
    def __init__(self, scoring_matrix_pth):
        self.sequence = sequence
        self.align_sequences = []
        self.pairwise_matrix = []
        self.scoring_matrix = pd.read_csv(scoring_matrix)


    def __call__(self, sequence):
        distance_matrix = create_distance_matrix()
        quide_tree = create_quide_tree(distance_matrix)
        pass
        


    def phylogenic_distance(ac_a,ac_b) -> bool:
        return ac_a and ac_b
    

    def similarity_reduce(acc: int, value: bool) -> int:
        return acc + int(value == True)

    
    def check_vals(self,i,j) -> int:
        if self.pairwise_matrix[0] == self.pairwise_matrix[0]:
            return 1 
        return -1

    
    def glob_ali_inner_func(self, i, j, d) -> list[int]:
        path += list(max([glob_ali_inner_func(i-1, j-1) + check_vals(i, j), glob_ali_inner_func(i-1, j) + d, glob_ali_inner_func(i, j-1) + d ]))
        return path
    

    def global_alignment(self, seq_a, seq_b) -> list[int]:
        d = 2
        seq_a_len = len(seq_a) - 1
        seq_b_len = len(seq_b) - 1
        
        self.pairwise_matrix = [[0] + seq_a.split()] + [[0] + [[x] + [None] * (seq_b_len) for x in seq_b]        
        self.pairwise_matrix[1] = [ (i+1)*d for i in range(seq_a_len)]
        self.pairwise_matrix = [row[1] = (i+1)*d for i,row in enumerate(self.pairwise_matrix)]
        return glob_ali_inner_func(seq_a_len, seq_b_len, d)    

    
        
    def browse_similar_ac(percentual_similarity: int, amino_acid_seq: str) -> list[int]:



    def create_distance_matrix(self) -> list[list[string]]:
        distance_matrix = []
        for i in range(len(self.align_sequences), 1, -1):
            seq = 
            distance_matrix.append([None]*len(self.align_sequences)-1)
            for j in range(0, i):
                similarity = list(map(phylogenic_distance, self.align_sequences[j]))
                distance_matrix[i][j] = reduce(similarity_reduce,similarity)
            
        return distance_matrix


    def create_quide_tree(self, distance_matrix):
        phylo_tree = PhylogenicTree()
        for i in range(len(distance_matrix) - 1):
            closest_match = -1
            for j in range(len(distance_matrix) - 1):
                if distance_matrix[i][closest_match] > distance_matrix[i][j]:
                    closest_match = j
            phylo_tree.seq_a = 

        pass


    def align_clusters(self)
        pass


if __name__ == "__main__":
    file_mode = FileMod("./data/raw/AMINtoSEC.csv")
