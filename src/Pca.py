import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


def pcaFunction(data_matrix, scree_plot_path, pca_plot_path):
    genes = ['gene' + str(i) for i in range(1, data_matrix.shape[0] + 1)]

    att = ['att' + str(i) for i in range(1, 11)]

    data = pd.DataFrame(data_matrix, columns=[*att], index=genes)

    print(data.head())
    print(data.shape)

    #########################
    #
    # Perform PCA on the data
    #
    #########################
    scaled_data = preprocessing.scale(data.T)

    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    #########################
    #
    # Draw a scree plot and a PCA plot
    #
    #########################

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot - attributes of poker hands of testing data')
    plt.savefig(scree_plot_path, dpi=400)
    plt.show()

    pca_df = pd.DataFrame(pca_data, index=[*att], columns=labels)

    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('PCA Graph - poker hands of testing data')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))

    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

    plt.savefig(pca_plot_path, dpi=400)
    plt.show()

    #########################
    #
    # Determine which genes had the biggest influence on PC1
    #
    #########################

    loading_scores = pd.Series(pca.components_[0], index=genes)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

    top_10_genes = sorted_loading_scores[0:10].index.values

    print(loading_scores[top_10_genes])