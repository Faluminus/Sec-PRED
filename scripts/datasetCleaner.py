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


def remove_duplicates(query: pd.Series, df: pd.DataFrame, ac_name: str, ss_name: str) -> pd.DataFrame:
    rows_to_drop = []  
    for i, v in df.iterrows():
        if v[ac_name] == query[ac_name]:
            rows_to_drop.append(i)
        elif v[ss_name] == query[ss_name]:  
            rows_to_drop.append(i)
    df = df.drop(rows_to_drop, axis='index').reset_index(drop=True)
    return df


def remove_similars(query: pd.Series, df: pd.DataFrame, max_similarity: float, ac_name: str, ss_name:str) -> pd.DataFrame:
    for i, val in df.iterrows():
        mached = 0
        unmached_ac = abs(len(query[0]) - len(val[ac_name]))
        unmached_ss = abs(len(query[1] - len(val[ss_name])))
        shorter_seq_len_ac = len(query[0])*(query[0] < val[ac_name]) + len(val[ac_name])*(query[0] > val[ac_name])
        shorter_seq_len_ss = len(query[1])*(query[1] < val[ss_name]) + len(val[ss_name])*(query[1] > val[ss_name])
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
    ac_name = "input"
    ss_name = "dssp8"
    df = pd.read_csv("./../data/raw/data.csv")
    df = normalize(df, 500, ac_name, ss_name)
    for _, row in df.iterrows():
        df = remove_duplicates(row, df, ac_name, ss_name)
    for _, row in df.iterrows():
        df = remove_similars(row, df, 80, ac_name, ss_name)
    df.to_csv("./..data/processed/AMINtoSECcleared.csv", index=False)
    
