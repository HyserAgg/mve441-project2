import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn stuff
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Local imports
from filters import VarianceFilter

def main():
    print("Reading data..")
    feature_df = pd.read_csv("data/data.csv")
    label_df = pd.read_csv("data/labels.csv")
    X = feature_df.iloc[:,1:].to_numpy()

    print("Performing PCA..")
    pca = PCA(n_components=0.8, svd_solver="full")
    X = pca.fit_transform(X)

    print("Performing clustering..")
    kmean = KMeans(n_clusters=5)
    l = kmean.fit_predict(X)
    print(l)


def create_preprocessor(thrshld = 0.5, percent_variance = 0.8):
    return Pipeline(steps=[
                        ('variance_filter', VarianceFilter(thrshld)),
                        ('scaler', RobustScaler()),
                        ('PCA', PCA(n_components=percent_variance, svd_solver = 'full'))])

if __name__ == "__main__":
    main()