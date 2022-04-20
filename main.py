
from http.client import NO_CONTENT
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from os.path import exists
from pyarrow import csv
from sklearn.preprocessing import MinMaxScaler, normalize,LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn import mixture as mx
from sklearn.datasets import make_blobs
# Classes
#--------------------------------------------------------------------------------------------------------#
class VarianceFilter():
    def __init__(self, thrshld):
        self.thrshld = thrshld

    def fit(self, X, y = None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_norm = RobustScaler().fit_transform(X.copy())
        X_var = X_norm.var(axis=0)
        X_ = X_copy[:,(X_var > self.thrshld)]
        return X_
    
class CorrelationFilter():
    def __init__(self, thrshld):
        self.thrshld = thrshld

    def fit(self, X, y = None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        i = 0
        while True: 
            _,m = np.shape(X_)
            print(m)
            remaining_cols = np.arange(i,m)
            # Find correlation with all columns j>i
            corr = np.ndarray(shape = np.shape(remaining_cols))
            corr = np.array([np.correlate(X_[:,i],X_[:,j]) for j in remaining_cols]).T
            corr[0] = 0
            #if corr > thrshld, then this feature is updated in the mask and isn't checked for i+1
            corr_bool = (corr > self.thrshld)
            print(np.shape(corr_bool))
            X_ = np.delete(X_, corr_bool, axis = 1)
            i += 1
        return X_

# General info
#--------------------------------------------------------------------------------------------------------#
    # Silhouette score: Aim towards 1
    # CH score: The largers the better since CH = (a*Separation)/(b*Cohesion)
    # DB score: Closer to zero is preferred
#--------------------------------------------------------------------------------------------------------#
def main():
    ## Models ## 
    max_clusters = 8
    n_inits = 5
    clusters = list(range(2,max_clusters))
    models = [KMeans(n_init = n_inits, init = 'k-means++', n_clusters=i) for i in clusters]
    gmodels = [mx.GaussianMixture(n_components = i,covariance_type="full", n_init = n_inits) for i in clusters]
    models.extend(gmodels)
    clusters.extend(clusters)
    ## Load data ##
    label_df, feature_df = load_SEQ_data();
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(label_df)
    #X = gen_test_data(800, 20000, 8, 0.6)

    ## Feature pre - vis ##
    #col_hist(feature_df,1000)

    ## Preprocessing ##
    preprocessor = Pipeline(steps=[
                        ('variance_filter', VarianceFilter(3.5e-4)),
                        ("scaler", RobustScaler()),
                        ("correlation_filter", CorrelationFilter(0.75)),
                        ('PCA', PCA(n_components=0.90, svd_solver = 'full')),
    ])
    X = feature_df.to_numpy() # dim = (801, 20531)
    X_ = preprocessor.fit(X).transform(X)
    
    ## Initial vizualisation ## 
    pca = preprocessor["PCA"]
    #scree_plot(pca,0.95)
    #pair_plot(pca, 5)

    ## Run clustering ##
    #metrics_df, predicted_labels = run_clustering(X_,models, clusters)
    
    #plot_metrics(metrics_df)

    ## Our clustering: Names are: GaussianMixture(n_components=%),KMeans(n_clusters=%, n_init=%)
    #model = models[2]
    #print(model)
    #pred_labels = predicted_labels[model]
    #pair_plot(X_, 5, pred_labels)
    ## "Best" metrics + visualize ##
    #pair_plot(X_, 5, true_labels)

    plt.show()
    
    
    # Write function that generates data dim (., 20531) with specified amount of classes and distinctness
    #plt.show()
    # Vizualise our predictions and compare with true labels, the different metrics, 
    # https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    # https://github.com/HCGrit/MachineLearning-iamJustAStudent/blob/master/PipelineFoundation/Pipeline_Experiment.ipynb
    # https://towardsdatascience.com/building-a-machine-learning-pipeline-3bba20c2352b
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py

def run_clustering(X_,models, clusters):
    # Run clustering and return the metrics and the predicted labels for each model
    predicted_labels = {}
    columns  = ['Clusters','Model','S', 'DB', 'CH']
    metrics_df = pd.DataFrame(columns = columns)
    model_names = [model.__class__.__name__ for model in models]
    no_models = len(model_names)
    no_metrics = len(columns)-2
    shape = (len(model_names),len(columns)-2)
    metrics_df['Clusters'] = pd.Series(clusters, dtype= int)
    metrics_df['Model'] = pd.Series(model_names, dtype= "string")
    metrics = np.ndarray(shape= shape)
    clustering_metrics = [silhouette_score, davies_bouldin_score, calinski_harabasz_score]
    for i,model in enumerate(models):
        model.fit(X_)
        try:
            pred_labels = model.labels_
        except AttributeError:
            pass
        try: 
            pred_labels = model.predict(X_)
        except AttributeError:
            pass

        metrics[i,:] = [m(X_,pred_labels) for m in clustering_metrics]
        predicted_labels[model] = pred_labels

    metrics_df['S'] = metrics[:,0]
    metrics_df['DB'] = metrics[:,1]
    metrics_df['CH'] = metrics[:,2]

    return (metrics_df, predicted_labels)

def load_SEQ_data() -> pd.DataFrame:
    if exists('./data.h5'):
        print("Hdf file accessible; reading")
        feature_df = pd.read_hdf('./data.h5')
        label_df = pd.read_csv('./labels.csv')
        label_df = label_df["Class"]
        feature_df = feature_df.iloc[:,1:]
        return label_df, feature_df
    else:
        print("No hdf file; reading csv and creating hdf")
        df = pd.read_csv("./data.csv")
        label_df = pd.read_csv('./labels.csv')
        label_df = label_df["Class"]
        df.to_hdf('data.h5',key = 'df', mode='w') 
        df = df.iloc[:,1:]  
        return label_df, feature_df

def col_hist(df: pd.DataFrame, no_bins: int) -> None:
    # Plots histograms of the different features
    df_mean = df.mean(axis=0).transpose()
    df_var = df.var(axis=0).transpose()

    colors = ['b','y']
    fig, axes = plt.subplots(1,2)
    mean_hist = df_mean.hist(bins = no_bins, ax=axes[0], color = colors[0])
    std_hist = df_var.hist(bins = no_bins, ax=axes[1], color = colors[1])
    plt.show()

def scree_plot(pca,thrshld):
    ratios = pca.explained_variance_ratio_
    ratio_sums = np.cumsum(ratios)
    print(ratio_sums)
    index = np.argmax(ratio_sums>thrshld)
    comps = np.arange(1,len(ratios)+1)
    ratios = ratios/np.max(ratios)
    plt.plot(comps, ratios)
    plt.legend("Ratio of explained variance, div by max")
    plt.plot(comps, ratio_sums)
    plt.legend("Cumulative sum of expl. var. in %")
    plt.vlines(index, 0,1,linestyles='dashed')
    plt.show()
def pair_plot(X,no_comps, pred_labels = None):
    X_ = X.copy()
    X_ = X_[:,:no_comps]
    labels = ["PC%s" % _ for _ in range(1,no_comps+1)]
    df = pd.DataFrame(X_,columns=labels)
    if not isinstance(pred_labels, type(None)):
        labels = pd.Series(labels)
        df["labels"] = pred_labels
        print(df)
        _ = sns.pairplot(df, hue = "labels")
    else: 
        _ = sns.pairplot(df)
    
    
def plot_metrics(metrics: pd.DataFrame):
    no_models = metrics['Model'].nunique()
    models = metrics['Model'].unique()
    metric_cols = metrics.iloc[:,2:].columns
    # Plot the different metrics
    no_cols = metrics.shape[1]-2
    fig, axs = plt.subplots(no_models, no_cols)
    for i,model in enumerate(models):
        model_metrics = metrics[metrics['Model']==model]
        clusters = model_metrics['Clusters']
        for j,col in enumerate(metric_cols):
            metric_col = model_metrics[col]

            axs[i,j].plot(clusters, metric_col)
            axs[i,j].set_title((model,col))

def gen_test_data(n_samples,n_features, n_centers,cluster_std ):
    X, _ = make_blobs(n_samples = n_samples,
                  n_features = n_features, 
                  centers = n_centers,
                  cluster_std = cluster_std,
                  center_box = (0,1),
                  shuffle = True)
    return X


if __name__ == "__main__":
    main()