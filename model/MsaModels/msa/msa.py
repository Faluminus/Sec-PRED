import pandas as pd

class PhylogenicTree():
    def __init__(self, seq_a: str, seq_b: str, parent: 'PhylogenicTree'):
        self.seq_a = seq_a
        self.seq_b = seq_b
        self.parent = parent
                
class Msa():
    def __init__(self, scoring_matrix, amino_acids):
        self.scoring_matrix = scoring_matrix
        self.amino_acids = amino_acids

    def __call__(self, query):
        similar_sequences = self.__get_similars(query)
        print(similar_sequences)

    def __get_similars(self, query: str) -> list[str]:
        similars = list()
        for row in self.amino_acids:
            matches = sum(1 for a, b in zip(row, query) if a == b)
            accuracy = (matches / max(len(row), len(query)))
            if accuracy > 0.8:
                similars.append(row)
        return similars 
    


if __name__ == "__main__":
    df = pd.read_csv('./../data/raw/AMINtoSEC.csv')
    blosum = pd.read_csv('./msa/blosum62.csv')
    msa = Msa(blosum,df['AminoAcidSeq'])
    msa("PVRVGLSVDASALGHTIPPDYTGLSYEQAQMANPNYFSGANTQLAGFLRTLGRQGVLRIGGNTSEYTFWNRHAKPTAADEHLAAGPDKGHHAAAREVITPEAVNNLSEFLDKTGWKLIYGLNLGKGTPENAADEAAYVMETIGADRLLAFQLGNEPDLFYRNGIRPASYDFAAYAGDWQRFFTAIRKRVPNAPFAGPDTAYNTKWLVPFADKFKHDVKFISSHYYAEGPPTDPSMTIERLMKPNPRLLGETAGLKQVEADTGLPFRLTETNSCYQGGKQGVSDTFAAALWAGDLMYQQAAAGSTGINFHGGGYGWYTPVAGTPEDGFIARPEYYGMLLFAQAGAGQLLGAKLTDNSAAPLLTAYALRGTDGRTRIALFNKNLDADVEVAISGVASPSGTVLRLEAPRADDTTDVTFGGAPVGASGSWSPLVQEYVPGHSGQFVLHMRKASGALLEFA")
