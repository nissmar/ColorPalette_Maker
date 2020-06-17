import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import colorsys

def imageToArray(path, debug = True, mode = "hsv"): #returns an array of 3D points in RGB or HSV
    if debug:
        print("Loading Image...")
    img = mpimg.imread(path)
    n,m,_ = (img.shape)
    if (mode == "hsv"):
        X = [colorsys.rgb_to_hsv(img[i//m][i%m][0]/255.0, img[i//m][i%m][1]/255.0, img[i//m][i%m][2]/255.0) for i in range(n*m)]
    else:
        X = [[img[i//m][i%m][0]/255.0,img[i//m][i%m][1]/255.0,img[i//m][i%m][2]/255.0] for i in range(n*m)]
    if debug:
        print(" ...[OK]")
    return X

def computeCluster(X,type = "Gaussian",n_components=6, debug = True): #Compute the cluster for a given array
    if debug:
        print("Computing cluster...")
    if (type == "Gaussian"):
        ms = GaussianMixture(n_components=n_components)
    elif (type == "BayesianGaussian"):
        ms = BayesianGaussianMixture(n_components=n_components,weight_concentration_prior=100)
    else:
        raise("ERROR : UNKNOWN TYPE")
    ms.fit(X)
    labels = ms.predict(X)
    n_clusters_ = len(ms.means_)
    cluster_centers_weights = [[ms.means_[i],ms.weights_[i]] for i in range(n_clusters_)]
    cluster_centers_weights.sort(key = lambda x: x[1])
    if debug:
        print(" ...[OK]")
    return cluster_centers_weights, labels

def computePalette(cluster_centers_weights,palette_limit=7, mode = "hsv"): #return a Palette in RGB format
    P = [[elem[0] for elem in cluster_centers_weights[:min(palette_limit,len(cluster_centers_weights))]]]
    if (mode == "hsv"):
        P[0].sort(key = lambda x : x[2])
        P = [[colorsys.hsv_to_rgb(elem[0],elem[1],elem[2]) for elem in P[0]]]
    else:
        P = [[[elem[col] for col in range(3)] for elem in P[0]]]
        P[0].sort(key = lambda x : colorsys.rgb_to_hsv(x[0],x[1],x[2])[2])
    return P

def showData(X, labels, cluster_centers_weights, mode = "hsv"): #Shows the datza 
    n_clusters_ = len(cluster_centers_weights)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if (mode == "hsv"):
        pass
    else:
        X = [colorsys.rgb_to_hsv(elem[0],elem[1],elem[2]) for elem in X]
    for k in range(n_clusters_):
        points = [i for i in range(len(labels)) if (labels[i] == k)]
        ax.plot([X[i][0] for i in points], [X[i][1] for i in points],[X[i][2] for i in points], '.')
    plt.show()


### public

def paletteFromImage(path, n_components=8, palette_limit=5, debug = False, mode = "hsv", clustering_type = "Gaussian", show_clustering = False):
    # n_components: number of cluster components
    # palette_limit: numbers of color in palette
    # debug: show log
    # mode: can be Hue Saturation Value or RGB
    # clustering type: can be Gaussian or BayesianGaussian
    # show_clustering: show the plot of the clusters
    X = imageToArray(path, mode=mode, debug=debug)
    cluster_centers_weights, labels = computeCluster(X,n_components=n_components, debug=debug, type = clustering_type)
    if show_clustering:
        showData(X, labels, cluster_centers_weights, mode=mode)
    P = computePalette(cluster_centers_weights,palette_limit=palette_limit,mode=mode)
    return P
