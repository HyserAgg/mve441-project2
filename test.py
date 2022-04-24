import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn stuff
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Local imports
from filters import VarianceFilter

def main():
    feature_df = pd.read_csv("data/data.csv")
    label_df = pd.read_csv("data/labels.csv")

    X = feature_df.iloc[:,1:].to_numpy()

    pca = PCA(n_components=0.8, svd_solver="full")
    X = pca.fit_transform(X)

    ratios = pca.explained_variance_ratio_
    ratio_sums = np.cumsum(ratios)
    comps = np.arange(1,len(ratios)+1)
    ratios = ratios/np.max(ratios)
    plt.plot(comps, ratios)
    plt.plot(comps, ratio_sums)
    plt.legend(["Ratio of explained variance, div by max","Cumulative sum of expl. var. in %"])
    plt.show()


def create_preprocessor(thrshld = 0.5, percent_variance = 0.8):
    return Pipeline(steps=[
                        ('variance_filter', VarianceFilter(thrshld)),
                        ('scaler', RobustScaler()),
                        ('PCA', PCA(n_components=percent_variance, svd_solver = 'full'))])

if __name__ == "__main__":
    main()