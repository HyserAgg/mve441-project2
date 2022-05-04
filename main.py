import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# General 'from' import
from os.path import exists
from sklearn import cluster
from statsmodels.distributions.empirical_distribution import ECDF
from typing import Any

# Scikit-learn stuff
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, fowlkes_mallows_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn import mixture as mx
from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy.stats import pearsonr

# Local import
from filter import UnimodalFilter, MultimodalFilter, VarianceFilter


def main():


    ## Question 1 ##
    q1_plot = {
            "hist_plot":    False,
            "scree_plot":   False,
            "metrics_plot": True, 
            "pair_plot":    True
            }

    question_1(q1_plot)
   
    ## Question 2 ##
    #question_2()
    
    ## Question 3 ## 

    plt.show()

def question_1(plot):
    ## Data ## 
    label_df, feature_df = load_SEQ_data();
    X = feature_df.to_numpy()
    preprocessor = pre_processing()
    X = preprocessor.fit_transform(X)
    true_labels = LabelEncoder().fit_transform(label_df)
    ## Models ## 
    n_inits = 1000
    clusters = list(range(3,8))
    cases = [
            (KMeans, "n_clusters", {"n_init": n_inits, "init": "k-means++"}),
            (mx.GaussianMixture, "n_components", {"n_init" : n_inits}),
            (AgglomerativeClustering, "n_clusters", {"affinity":"cosine","linkage": "average"})
            ]
    cl_metrics = [silhouette_score, davies_bouldin_score, calinski_harabasz_score]

    ## Feature pre - visualisation ##
    #if plot["hist_plot"]:
        # col_hist(feature_df,1000)
    ## Initial vizualisation ## 
    # if plot["scree_plot"]:
    #     scree_plot(preprocessor["PCA"])
    if plot["pair_plot"]:
        pair_plot(X, 3, {"True_labels": true_labels})

    ## Run clustering ##
    metrics_df, predicted_labels = run_clustering(X,cases, clusters, cl_metrics)
    if plot["metrics_plot"]:
        metrics_plot(metrics_df)
    for key in predicted_labels.keys():
        preds = predicted_labels[key]
        print(key,adjusted_rand_score(true_labels, preds), fowlkes_mallows_score(true_labels, preds))
    ## Visualize clustering:
    # model = models[15]
    # pred_labels = predicted_labels[model]
    # if plot["pair_plot"]:
    #     pair_plot(X, 3, {"Predicted_labels": pred_labels})
    #     pair_plot(X, 3, {"True_labels":true_labels})
    # print(f"Rand score for {model}: ",adjusted_rand_score(true_labels,pred_labels))
    ## "Best" metrics + visualize ##
    #pair_plot(X_, 5, true_labels)

    plt.show()
    return 

def question_2():
    _, feature_df = load_SEQ_data();
    n_inits = 100
    clusters = list(range(2,10))
    cases = [
            #(KMeans, "n_clusters", {"n_init": n_inits, "init": "k-means++"}),
           # (mx.GaussianMixture, "n_components", {"n_init" : n_inits}),
            (AgglomerativeClustering, "n_clusters", {"linkage": "ward"})
            ]
    preprocessor = pre_processing()
    ms = consensus_clustering(feature_df, cases, clusters, preprocessor, verbose = True)
    for _, m in ms.items():
        ecdf = ECDF(m.flatten())
        plt.plot(ecdf.x, ecdf.y)
    plt.legend(ms.keys())
    return

def question_3():
    # Model on which stability is tested on and metrics
    # to calculate on the resulting model
    model = KMeans(n_clusters=5, n_init=25)

    print("Reading data..")
    feature_df = pd.read_csv("data/data.csv")
    label_df = pd.read_csv("data/labels.csv")
    X = feature_df.iloc[:,1:].to_numpy()
    y = label_df["Class"].to_numpy()

    # Standardising data
    print("Standardising data..")
    X = RobustScaler().fit_transform(X)
    
    print("Filtering features..")
    X_filtered = []
    X_filtered = [("Unfiltered", X)]

    # Unimodality filtering
    pipeline7 = Pipeline(steps=[
        ("unimodality_filter", UnimodalFilter())
    ])
    pipeline8 = Pipeline(steps=[
        ("multiimodality_filter", MultimodalFilter())
    ])
    X_filtered.append( ("Unimodal filtered alpha = 0.05", pipeline7.fit_transform(X)) )
    X_filtered.append( ("Multimodal filtered alpha = 0.05", pipeline8.fit_transform(X)) )

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


    # PCA filtering
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
    
    metrics = {}
    for data in X_filtered:
        print(f"Computing cluster stability for {data[0]}..")
        c = consensus_matrix(data[1], model, 0.8, 50)
        c_flat = c.flatten()
        ecdf = ECDF(c_flat)
        plt.plot(ecdf.x, ecdf.y, label=data[0])


        # Training model on all data and predics labels
        print(f"Fitting and predicting labels for {data[0]}..")
        pred_labels = model.fit_predict(data[1])

        # Calculating some metrics
        print(f"Calculating metrics for {data[0]}..")
        metrics[data[0]] = {
            "PAC": ecdf(0.99) - ecdf(0.01),
            "silhouette_score": silhouette_score(data[1], pred_labels),
            "davies_bouldin_score": davies_bouldin_score(data[1], pred_labels),
            "calinski_harabasz_score": calinski_harabasz_score(data[1], pred_labels),
            "fowlkes_mallows_score": fowlkes_mallows_score(y, pred_labels),
            "adjusted_rand_score": adjusted_rand_score(y, pred_labels), 
        }        

    plt.legend()
    plt.title(model)
    plt.savefig(f"./figures/{model}_features_stability.png")
    print(json.dumps(metrics, indent=2))

def consensus_clustering(df, 
                      cases,
                   clusters,
                   pipeline,     
                     k = 100, 
                    p = 0.8, 
            verbose = False):
    """
    inputs:
           df        - dataframe
           cases     - initialized sklearn model
           pipeline  - the given preprocessing pipeline
           k         - number of subsamples
           p         - proportion of samples 
           Q         - tuple of ecdf thresholds
           ecdf_plot - whether to make an ECDF plot

    outputs:          
           PACs      - PAC_k values for specificed Q
           ECDFs     - empirical cumulative distribution for each cluster count
           Mk        - Connectivity matrix for each cluster count


    Aim: Test cluster stability for different variance thresholds, models and numbers of PCA components, 
         return the consensus matrix for use as a similarity matrix for Agg. Hier. Clustering
        

    """
    print("Testing stability")
    def update_numerator(numerator,j, labels, subsample_indices):
        """
        inputs:
               numerator   - matrix dim n*n; sum of connectivity matrices
               pred_labels - vector of dim ~0.8n of labels, remember that this is a subsample¨
               connect     - connectivity matrix for current subsample run

        output:            - added 1 to all (i,j) where labels(i) = labels(j)
            
        """    
        connect = np.zeros((n,n), dtype = float)
        for label, ind in zip(labels, subsample_indices):
            # For a given label and index we add one if the prediction of i,j is the same
            active_subsample_indices = subsample_indices[np.nonzero(labels == label)]   # The rows of X where pred. is label
            connect[ind,active_subsample_indices] = 1.0  
        np.add(numerator[:,:,j], connect, numerator[:,:,j])
        return numerator

    def update_denominator(denominator,j, subsample_indices):
        """
        inputs: 
                denominator - matrix dim n*n; sum of indication matrices
                ss_indices  - vector of dim ~pn of choses indices for current subsample
                indicator   - indicator matrix for current subsample

        output:             - added ones to all (i,j) in ss_indices 
            
        """
        indicator = np.zeros((n,n), dtype = float)
        for ind in subsample_indices:          
            indicator[ind,subsample_indices] = 1.0            
        np.add(denominator[:,:,j], indicator, denominator[:,:,j])      # Sum the indicator matrices
        return denominator


    models = list()
    # Create the list of models
    for model, comp_param, params in cases:
        for i in clusters:
            params[comp_param] = i
            models.append(model(**params))
            
    X = df.to_numpy()
    X = pipeline.fit_transform(X)
    n,_ = np.shape(X)

    consensus_matrices = {}

    no_models = len(models)
    numerator = np.zeros((n,n,no_models), dtype = float)     # We calculate the consensus matrix from the numerator 
    denominator = np.zeros((n,n,no_models), dtype = float)   # and denominator matrices i.e the sums of the connectivity 
                                                             # and identicator matrices, respectively
    subsample_size = int(np.floor(p*n))  
    for j in range(k):
        # For a given subsample, we update the denominator,  we fit all models and update the numerator                                    
        subsample_indices = np.random.choice(n, subsample_size, replace = False)
        X_ = X[subsample_indices,:]                          # Our subsampled data
        if verbose:
            print(f'At iteration: {j}')                
        for i, model in enumerate(models):
            model_ = clone(model)                            # Clone model and if possible set a random_state
 
            model_.fit(X_)                                   # Fit the clusterer to the data
            try: 
                pred_labels = model_.labels_
            except AttributeError:
                try: 
                    pred_labels = model_.predict(X_)
                except AttributeError:
                    print("Model has neither predict nor labels_ attr.")   
                    raise AttributeError

            #  Now to update the numerator and denominator matrices
            numerator   = update_numerator(numerator,i, pred_labels,subsample_indices)
            denominator = update_denominator(denominator,i,subsample_indices)

            assert(np.all(numerator<=denominator))
            
    assert(np.all(denominator != 0))
    consensus = np.divide(numerator, denominator)
    for i, model in enumerate(models):
        consensus_matrices[model] = consensus[:,:,i]
    
    return consensus_matrices

def run_clustering(X_,cases, clusters, cl_metrics):
    """
    inputs: 
            X_               - data arrays, preprocessed
            cases            - dict of models and their parameters
            clusters         - list of clusters
            cl_metrics       - list of metric objects
    
    
    output: 
            metrics_df       -  dataframe of metrics for a certain model and cluster
            predicted_labels - dict of the predictions
    """
    
    predicted_labels = {}
    models = list()
    model_names = list()
    clusters_ =  list()
    columns  = ['Clusters','Model']
    columns.extend(str(metric.__name__) for metric in cl_metrics)
    
    # Create the list of models and clusters
    for model, comp_param, params in cases:
        for i in clusters:
            params[comp_param] = i
            models.append(model(**params))
            model_names.append(model.__name__)
        clusters_.extend(clusters)

    # Initialize dataframe with clusters and model names
    metrics_df = pd.DataFrame(columns = columns)
    shape = (len(model_names),len(columns)-2)
    metrics_df['Clusters'] = pd.Series(clusters_, dtype= int)
    metrics_df['Model'] = pd.Series(model_names, dtype= "string")
    metrics = np.ndarray(shape)

    # Fit the model to the data and return the predicted clusters, evaluate these using the specified clustering metrics
    for i,model in enumerate(models):
        model.fit(X_)
        pred_labels = predict_cluster(model, X_)

        metrics[i,:] = [m(X_,pred_labels) for m in cl_metrics]
        predicted_labels[model] = pred_labels

    # Add our metrics to the dataframe
    for i,metric in enumerate(cl_metrics):
        metrics_df[metric.__name__] = metrics[:,i]
    return (metrics_df, predicted_labels)

def pre_processing(thrshld = 2.2, percent_variance = 0.8):
    # Return our preprocessing pipeline
    preprocessor = Pipeline(steps=[
                        ('variance_filter', VarianceThreshold(thrshld)),
                        ('scaler', MinMaxScaler()),
                        ('PCA', PCA(n_components=percent_variance, svd_solver = 'full'))])
    return preprocessor

def load_SEQ_data():
    # Reading .h5 files seem faster in this case
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

def col_hist(df: pd.DataFrame, no_bins: int):
    # Plots histograms of the different features
    fontsize = 30
    plt.figure()
    df_mean   = df.mean(axis=0).transpose()
    df_var    = df.var(axis=0).transpose()
    ecdf_mean = ECDF(df_mean)
    ecdf_var  = ECDF(df_var)
    plt.subplot(221)
    plt.hist(df_mean, bins = no_bins)
    plt.title("Feature means", fontdict={'fontsize': fontsize})

    plt.subplot(222)
    plt.hist(df_var, bins = no_bins)
    plt.title("Feature variances", fontdict={'fontsize': fontsize})

    plt.subplot(223)
    plt.plot(ecdf_mean.x, ecdf_mean.y)
    plt.title("ECDF of feature means",fontdict={'fontsize': fontsize})

    plt.subplot(224)
    plt.plot(ecdf_var.x, ecdf_var.y)
    plt.title("ECDF of feature variances",fontdict={'fontsize': fontsize})

def scree_plot(pca):
    fontsize = 25
    ratios = pca.explained_variance_ratio_
    ratio_sums = np.cumsum(ratios)
    comps = np.arange(1,len(ratios)+1)
    ratios = ratios/np.max(ratios)
    plt.plot(comps, ratios)
    plt.plot(np.concatenate(([0],comps)), np.concatenate(([0],ratio_sums)))
    plt.legend(["Standardised explained variance","Cumulative percentage explained variance"], fontsize = fontsize)
    plt.xlabel("No. of PCA components", fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    return

def pair_plot(X,no_comps, labels = None):
    """
    labels is a dict with the name we want in the plot
    """
    sns.set(font_scale = 2)
    X_ = X[:,:no_comps]
    cols = ["PC%s" % _ for _ in range(1,no_comps+1)]
    df = pd.DataFrame(X_,columns=cols)
    if not isinstance(labels, type(None)):
        name = list(labels)[0]
        df["labels"] = labels[name]
        g = sns.pairplot(df, hue = "labels")
        g.map_lower(sns.kdeplot, levels=4)
        if name == "True_labels":
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle("True labels")
        elif name == "Predicted_labels":
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle("Predicted labels")
        else:
            pass
    else: 
        g = sns.pairplot(df)
        g.title(name)
    return
    
def predict_cluster(model, X):
    try: 
        prediction = model.labels_
    except AttributeError:
        try: 
            prediction = model.predict(X)
        except AttributeError:
            print("Model has neither predict nor labels_ attr.")   
            raise AttributeError
    return prediction
 
def metrics_plot(metrics: pd.DataFrame):
    fig = plt.figure(constrained_layout=True)
    no_models = metrics['Model'].nunique()
    models = metrics['Model'].unique().to_numpy()
    metric_cols = metrics.iloc[:,2:].columns
    # Plot the different metrics
    no_cols = metrics.shape[1]-2
    # Create a row of figures for each model
    subfigs = fig.subfigures(nrows=no_models, ncols=1)
    # Iterate over subfigs & models
    for row, subfig in enumerate(subfigs):
        model = models[row]
        subfig.suptitle(f'{model}')
        model_metrics = metrics[metrics['Model']==model]
        clusters = model_metrics['Clusters']
        axs = subfig.subplots(nrows=1, ncols=no_cols)
        for i, ax in enumerate(axs):
            col = metric_cols[i]
            metric_col = model_metrics[col]
            ax.plot(clusters, metric_col)
            ax.set_title(f'{col}')

def consensus_matrix(X: np.ndarray, model: Any, p: float, M: int, verbose=False) -> np.ndarray:
    """
    Calculates the concensus matrix for the specifiend model on M different random sub-samples
    of X. 
    """
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

    n = len(X)
    connectivity_matrix   = np.zeros((n,n), dtype = float)         # We calculate the consensus matrix from the numerator 
    indicator_matrix = np.zeros((n,n), dtype = float)         # and denominator matrices i.e the sums of the connectivity 
    consensus_matrix   = np.zeros((n,n), dtype = float)         # and identicator matrices, respectively
                                                
    subsample_size = int(np.floor(p*n))                
    for i in range(M):
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
        connectivity_matrix = update_numerator(connectivity_matrix, pred_labels, subsample_indices)
        indicator_matrix = update_denominator(indicator_matrix, subsample_indices)

        assert(np.all(connectivity_matrix<=indicator_matrix))
        
    assert(np.all(indicator_matrix != 0))
    np.divide(connectivity_matrix, indicator_matrix, consensus_matrix, dtype = float)

    return consensus_matrix

if __name__ == "__main__":
    main()