from argparse import ArgumentError
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# General 'from' import
from os.path import exists
from sklearn import cluster
from statsmodels.distributions.empirical_distribution import ECDF
from typing import Tuple

# Scikit-learn stuff
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn import mixture as mx
from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy.stats import pearsonr

# Local import
from filters import VarianceFilter, CorrelationFilter

def main():
    ## Load data ##
    feature_df = pd.read_csv("data/data.csv").iloc[:,1:]
    label_df = pd.read_csv("data/labels.csv")["Class"]

    ## Question 1 ##
    #q1_plot = {
    #        "hist_plot":    False,
    #        "scree_plot":   False,
    #        "metrics_plot": False, 
    #        "pair_plot":    False
    #        }
   # question_1(X_,preprocessor, true_labels, q1_plot)
   
    ## Question 2 ##
    n_inits = 25
    q2_clusters = range(2,8)
    models = [KMeans(n_init = n_inits, init = 'k-means++', n_clusters=i) for i in q2_clusters]
    PACs = question_2(feature_df, models)
    print(PACs)
    plt.savefig("./figures/concensus_ecdf.png")

    ## Question 3 ##
    #question_3()

def question_1(X_, feature_df, label_df, plot):
    ## Models ## 
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(label_df)
    preprocessor = pre_processing()
    n_inits = 5
    clusters = list(range(2,8))
    models = [KMeans(n_init = n_inits, init = 'k-means++', n_clusters=i) for i in clusters]
    gmodels = [mx.GaussianMixture(n_components = i,covariance_type="full", n_init = n_inits) for i in clusters]
    models.extend(gmodels)
    clusters.extend(clusters)
    ## Feature pre - visualisation ##
    if plot["hist_plot"]:
        col_hist(feature_df,1000)
    ## Initial vizualisation ## 
    pca = preprocessor["PCA"]
    if plot["scree_plot"]:
        scree_plot(pca,0.75)
    if plot["pair_plot"]:
        pair_plot(pca, 5)

    ## Run clustering ##
    metrics_df, predicted_labels = run_clustering(X_,models, clusters)
    if plot["metrics_plot"]:
        metrics_plot(metrics_df)

    ## Our clustering:
    model = models[3]
    pred_labels = predicted_labels[model]
    if plot["pair_plot"]:
        pair_plot(X_, 5, pred_labels)
        pair_plot(X_, 5, true_labels)
    ## "Best" metrics + visualize ##
    #pair_plot(X_, 5, true_labels)

    plt.show()


def question_2(feature_df, 
                    models, 
                    k = 100, 
                    p = 0.8, 
                    Q = (0.01, 0.99), 
                    verbose = False):
    """
    inputs:
           X     - numpy array of dim = (samples, no.pca.comps)
           model - initialized sklearn model
           k     - number of subsamples
           p     - proportion of samples 
           Q     - tuple of ecdf thresholds
    output:      - PAC_k value for specificed Q

    TO DO: run several models, plot ecdf and return PAC for all of them (reuse plot_metrics), be able to run with different prepr. settings

    """
    print("Testing stability")
    def update_numerator(numerator, labels, subsample_indices):
        """
        inputs:
               numerator   - matrix dim n*n; sum of connectivity matrices
               pred_labels - vector of dim ~0.8n of labels, remember that this is a subsample¨
               connect     - connectivity matrix for current subsample run

        output:            - added 1 to all (i,j) where labels(i) = labels(j)
            
        """    
        connect = np.zeros((n,n), dtype = float)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            active_indices = np.nonzero(labels == label)                   # The indeces of where pred_labels equals label 
            active_subsample_indices = subsample_indices[active_indices]   # The rows of X where pred. is label
            for i, ind in enumerate(active_subsample_indices):      
                connect[ind,active_subsample_indices[i:]] = 1.0            # For a given active index i we do : d(i,j) & d(j,i) += 1 
                connect[active_subsample_indices[i:],ind] = 1.0            # for all j coming after i in active.ss. indices
        np.add(numerator, connect, numerator)                              # Sum the connectivity matrices         
        return numerator

    def update_denominator(denominator, subsample_indices):
        """
        inputs: 
                denominator - matrix dim n*n; sum of indication matrices
                ss_indices  - vector of dim ~pn of choses indices for current subsample
                indicator   - indicator matrix for current subsample

        output:             - added ones to all (i,j) in ss_indices 
            
        """
        indicator = np.zeros((n,n), dtype = float)
        unique_indices = np.unique(subsample_indices)
        for i, ind in enumerate(unique_indices):          
            indicator[ind,unique_indices[i:]] = 1.0      # We choose an index i, and for all other subsampled indeces j 
            indicator[unique_indices[i:],ind] = 1.0      # we set d(i,j) and d(j,i) to one
        np.add(denominator, indicator, denominator)      # Sum the indicator matrices
        return denominator

    PACs = {}
    
    X = feature_df.to_numpy()
    preprocessor = pre_processing()
    X = preprocessor.fit_transform(X)
    n,m = np.shape(X)
    for model in models:
        numerator   = np.zeros((n,n), dtype = float)         # We calculate the consensus matrix from the numerator 
        denominator = np.zeros((n,n), dtype = float)         # and denominator matrices i.e the sums of the connectivity 
        consensus   = np.zeros((n,n), dtype = float)         # and identicator matrices, respectively
                                                    
        subsample_size = int(np.floor(p*n))                
        for i in range(k):
            model_ = clone(model)                            # Clone model and if possible set a random_state
            try: 
                s = np.random.randint(1e3)
                model_.random_state = s
                if verbose:
                    print(f'At iteration {i}, with random state {s}')
            except AttributeError: 
                print("No random state attribute")
                pass
            subsample_indices = np.random.randint(low = 0, high = n, size = subsample_size)
            X_ = X[subsample_indices,:]                      # Our subsampled data   
            model_.fit(X_)                                   # Fit the clusterer to the data
            try:                                             # Return the predicted labels
                pred_labels = model_.labels_
            except AttributeError:
                try: 
                    pred_labels = model_.predict(X_)
                except AttributeError:
                    print("Model has neither predict nor labels_ attr.")   
                    raise AttributeError

            #  Now to update the numerator and denominator matrices
            numerator    = update_numerator(numerator, pred_labels,subsample_indices)
            denominator  = update_denominator(denominator,subsample_indices)

            assert(np.all(numerator<=denominator))
            
        assert(np.all(denominator != 0))
        np.divide(numerator, denominator, consensus, dtype = float)
        consensus = consensus.flatten()     # Consensus is now a 1D vector of (hopefully) mostly ones and zeros

        ecdf = ECDF(consensus)              # The empirical cumulative distribution function
        plt.plot(ecdf.x, ecdf.y)
        PACs[model] = ecdf(Q[1])-ecdf(Q[0]) # PAC values for the different models
    plt.legend(models)
    return PACs


def question_3():
    print("Reading data..")
    feature_df = pd.read_csv("data/data.csv")
    label_df = pd.read_csv("data/labels.csv")
    X = feature_df.iloc[:,1:].to_numpy()
    
    print("Filtering features..")
    X_filtered = [("Unfiltered", X)]

    # Variance feature filtering
    pipeline1 = Pipeline(steps=[
        ("variance_filter", VarianceFilter(0.8))
    ])
    pipeline2 = Pipeline(steps=[
        ("variance_filter", VarianceFilter(0.5))
    ])
    pipeline3 = Pipeline(steps=[
        ("variance_filter", VarianceFilter(0.3))
    ])
    X_filtered.append( ("Variance filtered 0.8", pipeline1.fit_transform(X)) )
    X_filtered.append( ("Variance filtered 0.5", pipeline2.fit_transform(X)) )
    X_filtered.append( ("Variance filtered 0.3", pipeline3.fit_transform(X)) )

    # TODO: PCA feature filtering
    pipeline4 = Pipeline(steps=[
        ("PCA", PCA(n_components=1))
    ])
    pipeline5 = Pipeline(steps=[
        ("PCA", PCA(n_components=10))
    ])
    pipeline6 = Pipeline(steps=[
        ("PCA", PCA(n_components=50))
    ])
    X_filtered.append( ("PCA with 1 component", pipeline4.fit_transform(X)) )
    X_filtered.append( ("PCA with 10 component", pipeline5.fit_transform(X)) )
    X_filtered.append( ("PCA with 50 component", pipeline6.fit_transform(X)) )
    
    for data in X_filtered:
        print(f"Computing cluster stability for {data[0]}..")
        c = consensus_matrix(data[1], "K-Means", 5, 0.8, 20)
        c_flat = c.flatten("C")

        print("Plotting eCDF")
        ecdf = ECDF(c_flat)
        plt.plot(ecdf.x, ecdf.y, label=data[0])
    plt.legend()
    plt.show()


def run_clustering(X_,models, clusters):
    # Run clustering and return the metrics and the predicted labels for each model
    predicted_labels = {}
    columns  = ['Clusters','Model','S', 'DB', 'CH']
    metrics_df = pd.DataFrame(columns = columns)
    model_names = [model.__class__.__name__ for model in models]
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

def pre_processing(thrshld = 0.5, percent_variance = 0.8):
    preprocessor = Pipeline(steps=[
                        ('variance_filter', VarianceFilter(thrshld)),
                        ('scaler', RobustScaler()),
                        ('PCA', PCA(n_components=percent_variance, svd_solver = 'full'))])
    return preprocessor

def load_SEQ_data() -> pd.DataFrame:
    if exists('data/data.h5'):
        feature_df = pd.read_hdf('data/data.h5')
        label_df = pd.read_csv('data/labels.csv')
        feature_df = feature_df.iloc[:,1:]
        return label_df["Class"], feature_df.iloc[:,1:]  
    else:
        feature_df = pd.read_csv("data/data.csv")
        label_df = pd.read_csv('data/labels.csv')
        feature_df.to_hdf('data/data.h5',key = 'df', mode='w') 
        return label_df["Class"], feature_df.iloc[:,1:]  

def col_hist(df: pd.DataFrame, no_bins: int) -> None:
    # Plots histograms of the different features
    df_mean = df.mean(axis=0).transpose()
    df_var = df.var(axis=0).transpose()
    _, axes = plt.subplots(1,2)
    df_mean.hist(bins = no_bins, ax=axes[0], color = 'b')
    df_var.hist(bins = no_bins, ax=axes[1], color = 'y')
    plt.show()

def scree_plot(pca,thrshld):
    ratios = pca.explained_variance_ratio_
    ratio_sums = np.cumsum(ratios)
    comps = np.arange(1,len(ratios)+1)
    ratios = ratios/np.max(ratios)
    plt.plot(comps, ratios)
    plt.plot(comps, ratio_sums)
    plt.legend(["Ratio of explained variance, div by max","Cumulative sum of expl. var. in %"])
    plt.show()

def pair_plot(X,no_comps, labels = None):
    X_ = X.copy()
    X_ = X_[:,:no_comps]
    cols = ["PC%s" % _ for _ in range(1,no_comps+1)]
    df = pd.DataFrame(X_,columns=cols)
    if not isinstance(labels, type(None)):
        df["labels"] = labels
        _ = sns.pairplot(df, hue = "labels")
    else: 
        _ = sns.pairplot(df)
    
 
def metrics_plot(metrics: pd.DataFrame):
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
    plt.show()


def consensus_matrix(X: np.ndarray, model_name: str, n_clusters: int, subsample_frac: float, M: int) -> np.ndarray:
    """
    Calculates the concensus matrix for the specifiend model on M different random sub-samples
    of X. 
    """
    # Initiates the different models corresponding to the different 
    # number of clusters
    model = None
    if model_name == "K-Means":
        model = KMeans(n_clusters=n_clusters)
    elif model_name == "Gaussian Mixture":
        model = mx.GaussianMixture(n_components=n_clusters, n_init = 10)
    else:
        raise ArgumentError(f"cluster stability calculation for {model_name} is not defined.")

    idx_list = [ix for ix in range(len(X))]
    m = np.zeros((len(X),len(X)), dtype=float)    
    j = np.zeros((len(X),len(X)), dtype=float)
    for jx in range(M):
        # Creates random sub-sample of data and fits model
        # and predicts the cluster index for each of the sub-sampled
        # observations
        X_subsample_idx = np.random.choice(idx_list, size=int(np.floor(subsample_frac*len(X))), replace=False)
        cluster_idx = model.fit_predict(np.take(X, X_subsample_idx, axis=0))

        # Calculates connectivity and indicator matrix using above sub-sample
        m += connectivity_matrix(cluster_indicees=cluster_idx, sample_indicees=X_subsample_idx, max_samples=len(X)).astype("float64")
        j += indicator_matrix(sample_indicees=X_subsample_idx, max_samples=len(X)).astype("float64")
    
    # Calculates and returns the consensus matrix
    c = np.divide(m,j)
    return c


def connectivity_matrix(cluster_indicees: np.ndarray, sample_indicees: np.ndarray, max_samples: int) -> np.ndarray:
    """
    Calculates the connectivity matrix from a list of cluster indicees, list of the
    observation indicees and the maximum number of samples, i.e. (largest sample index)+1.
    """
    m = np.zeros((max_samples, max_samples), dtype=int)
    for ix in range(len(cluster_indicees)):
        for jx in range(len(cluster_indicees)):
            if cluster_indicees[ix] == cluster_indicees[jx]:
                m[sample_indicees[ix],sample_indicees[jx]] = 1

    return m


def indicator_matrix(sample_indicees: np.ndarray, max_samples: int) -> np.ndarray:
    """
    Calculates the indicator matrix from a list of sample indicees together
    with the maximum number of samples, i.e. (largest sample index)+1.
    """
    m = np.zeros((max_samples, max_samples), dtype=int)
    for ix in range(max_samples):
        for jx in range(max_samples):
            if ix in sample_indicees and jx in sample_indicees:
                m[ix,jx] = 1

    return m


if __name__ == "__main__":
    main()