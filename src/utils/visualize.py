from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from cuml.manifold import TSNE # If available, use it or
# from tsnecuda import TSNE # If available, use it
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import os

def visualize_latent_space(x, labels, n_clusters, range_lim=(-80, 80), perplexity=40, is_save=False, save_path=None):
    # tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=1000, init='random')
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(x)
    df_subset = pd.DataFrame()
    
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    df_subset['Y'] =  labels
    
    n_comps = len(np.unique(labels).tolist())
    
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='Y',
        palette=sns.color_palette(n_colors=n_comps),
        data=df_subset,
        legend="full",
        alpha=0.3
    ).set(xlim=range_lim,ylim=range_lim)
    
    if is_save:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        save_path = save_path if save_path else ''
        plt.savefig(save_path)
        plt.close('all')
        img = Image.open(save_path)
        return img
    return None