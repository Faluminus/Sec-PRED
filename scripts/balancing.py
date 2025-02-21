import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def counter(arr):
    ss = {}
    for row in arr:
        for e in row:
            if e not in ss:
                ss[e] = 0
            ss[e] += 1
    return ss


def balancer(counts: dict, df: pd.DataFrame):
    sorts = {}
    for key in counts.keys():
        array = df['SecondaryStructureSeq']
    pass
            

def x_fraction(seq):
    if len(seq) == 0:
        return 0
    return seq.count('X') / len(seq)

if __name__=='__main__':
    threshold = 0.1
    dataset = "./../data/processed/AMINtoSECcleared.csv"
    df = pd.read_csv(dataset)
    df = df[df['Kelvin'] > 300]
    print(len(df))
    #df['x_fraction'] = df['AminoAcidSeq'].apply(x_fraction)
    #df_filtered = df[df['x_fraction'] <= threshold].copy()
    #short_sequences = short_sequences[short_sequences['SecondaryStructureSeq'].apply(len) < 1000]
    #df_filtered.to_csv("./../data/processed/AMINtoSECwithX_fraction.csv", index=False)
    #print(df_filtered.head())