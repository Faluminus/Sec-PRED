import pandas as pd
from Bio.Data import IUPACData



if __name__ == '__main__':
    three_to_one = IUPACData.protein_letters_3to1.copy()
    path = '../data/raw/PKAD2_DOWNLOAD.xlsx'
    df = pd.read_excel(path)
    df = df.drop(['Chain', 'Res. ID', 'Expt. Uncertainty', 'Neighbors <4A', '# HB R-R', '#HB R(D)-BB', '# HB  R-BB(D)', '#HB R(A)-BB', '# HB R-BB(A)', '#HB BB-BB', '%SASA', 'Expt. Method', 'Expt. Salt Concentration', 'Expt. Temp.', 'Reference'], axis=1)
    df = df.dropna()
    df['pH_high'] = [x.split('-')[1] for x in df['Expt. pH']]
    df['pH_low'] = [x.split('-')[0] for x in df['Expt. pH']]
    df['Res. Name'] = [x.split()[1] for x in df['Res. Name']]
    df = df.drop('Expt. pH', axis=1)
    print(df.head())
    df.to_csv('../data/processed/PKAD2_CLEANED.csv')