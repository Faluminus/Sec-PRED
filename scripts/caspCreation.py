import pandas as pd

df = pd.read_csv('./../data/raw/CASP15.csv')

final_data = {'input': [], 'dssp8': []}
current_chain = df.iloc[0][2]  
ac_chain = []
ss_chain = []
for index, row in df.iterrows():
    if row[2] == current_chain:
        ac_chain.append(row[1])
        ss_chain.append(row[4])
    else:
        if ''.join(ac_chain) != '!':
            final_data['input'].append(''.join(ac_chain))
            final_data['dssp8'].append(''.join(ss_chain))
        ac_chain = [row[1]]
        ss_chain = [row[4]]
        current_chain = row[2]
final_data['input'].append(''.join(ac_chain))
final_data['dssp8'].append(''.join(ss_chain))

df = df.from_dict(final_data)
df.to_csv('./../data/processed/CASP14.csv',index=False)
