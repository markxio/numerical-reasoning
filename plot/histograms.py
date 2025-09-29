import tiktoken
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    prepare_plot("../dataset_context_long.csv", "context")
    prepare_plot("../dataset_context_long.csv", "short_context")

def prepare_plot(csv_file, col):
    golden = pd.read_csv(csv_file, sep=";")

    tokens_list = []
    n_list = []
    for sentence in tqdm(golden[col].tolist()):
        tokens = get_tokens(sentence)
        tokens_list.append(tokens)
        n_list.append(len(tokens))
    
    print(f"unique bins: {len(list(set(n_list)))}")
    plot_hist(n_list, col)

def plot_hist(bins, col):
    arr = np.array(bins)
   
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    ax.hist(bins, bins=317)

    #fig = plt.hist(bins)
    ax.set_ylabel(f"Number of {col}s")
    ax.set_xlabel("Number of tokens")
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.axvline(arr.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.savefig(f"dataset_{col}_histogram_bins_317.png")
    plt.savefig(f"dataset_{col}_histogram_bins_317.pdf")

    print("------------------------------")
    print(f"{col}:")
    print(f"min: {arr.min()}")
    print(f"max: {arr.max()}")
    print(f"mean: {arr.mean()}")     
    print(f"median: {np.median(arr)}")

def get_tokens(text):
    # Select the encoding model (e.g., "gpt-3.5-turbo")
    #encoding = tiktoken.get_encoding("gpt-4o")
    encoding = tiktoken.get_encoding("o200k_base") # google
    tokens = encoding.encode(text)
    return tokens

if __name__=="__main__":
    main()
