import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from collections import Counter

def normalize(df: pd.DataFrame, max_seq_len: int, ac_name: str, ss_name: str) -> pd.DataFrame:
    for i, row in df.iterrows():
        df.at[i, ac_name] = ''.join(
            [row[ac_name][j] if j < len(row[ac_name]) else '0' for j in range(max_seq_len)]
        )
        df.at[i, ss_name] = ''.join(
            [row[ss_name][j] if j < len(row[ss_name]) else '0' for j in range(max_seq_len)]
        )
    return df


def remove_similars(query: pd.Series, df: pd.DataFrame, max_similarity: float, ac_name: str, ss_name:str) -> pd.DataFrame:
    for i, val in df.iterrows():
        mached = 0
        unmached_ac = abs(len(query[ac_name]) - len(val[ac_name]))
        unmached_ss = abs(len(query[ss_name]) - len(val[ss_name]))
        shorter_seq_len_ac = len(query[ac_name])*(len(query[ac_name]) <= len(val[ac_name])) + len(val[ac_name])*(len(query[ac_name]) > len(val[ac_name]))
        shorter_seq_len_ss = len(query[ss_name])*(len(query[ss_name]) <= len(val[ss_name])) + len(val[ss_name])*(len(query[ss_name]) > len(val[ss_name]))
        for j in range(shorter_seq_len_ac-1):
            if val[ac_name][j] == query[ac_name]:
                mached += 1
        percentage = (100/(shorter_seq_len_ac + unmached_ac)) * mached
        if percentage >= max_similarity:
            df.drop(i, axis='index')
        mached = 0
        for j in range(shorter_seq_len_ss-1):
            if val[ss_name][j] == query[ss_name]:
                mached += 1
        percentage = (100/(shorter_seq_len_ss + unmached_ss)) * mached
        if percentage >= max_similarity:
            df.drop(i, axis='index')

if __name__ == '__main__':
    ac_name = "AminoAcidSeq"
    ss_name = "SecondaryStructureSeq"
    df = pd.read_csv("./../data/raw/AMINtoSEC.csv", on_bad_lines="skip")
    print(len(df))
    df = df.dropna()
    print(len(df))
    #df = df.dropna(subset=[ac_name])
    df = df[~df['Kelvin'].str.contains('NULL', na=True)]
    df = df[~df['Kelvin'].str.contains(';', na=True)]
    df['PH'] = pd.to_numeric(df['PH'], errors='coerce')
    df_clean = df.dropna(subset=['PH'])
    print(len(df))
    ln = len(df) -1
    #print(df[ss_name].value_counts())
    #df = normalize(df, 1000, ac_name, ss_name)
    #print(len(df))
    df = df.drop_duplicates(subset=[ss_name])
    print(len(df))
    
    df.to_csv("./../data/processed/AMINtoSECcleared.csv", index=False)
    #for i, row in df.iterrows():
    #    if i == ln:
    #        break
    #    if row is None:
    #        print("whoops")
    #    else:
    #        df = remove_similars(row, df, 80, ac_name, ss_name)
    #df.to_csv("./../data/processed/AMINtoSECcleared.csv", index=False)
    
