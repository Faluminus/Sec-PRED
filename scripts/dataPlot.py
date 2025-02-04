import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

df = pd.read_csv("./../data/raw/data.csv")

def freg(df: pd.DataFrame):
    class_count = Counter(''.join(df['dssp8']))


    secondary_structure = list(class_count.keys())
    frequencies = list(class_count.values())
    # Plotting
    plt.figure(figsize=(10, 6))  # Size of the figure
    plt.bar(secondary_structure, frequencies, color='skyblue')  # Create a bar plot

    # Adding labels and title
    plt.xlabel('Amino Acids')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of Amino Acids')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Show the plot
    plt.tight_layout()
    plt.show()

def seqlens(df: pd.DataFrame):
    unique_lens_ac = map()
    unique_lens_ss = map()
    for v in df:
        unique_lens_ac[len(v['AminoAcidSeq'])] += 1
        unique_lens_ss[len(v['SecondaryStructureSeq'])] += 1
        plt.figure(figsize=(10, 6))  # Size of the figure
        plt.bar(unique_lens_ac, color='skyblue')  # Create a bar plot

        # Adding labels and title
        plt.xlabel('Amino Acids')
        plt.ylabel('Frequency')
        plt.title('Frequency Distribution of Amino Acids')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

        # Show the plot
        plt.tight_layout()
        plt.show()


freg(df)