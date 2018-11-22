import numpy as np
from structuredlearning import Alpha, Loss
import matplotlib.pyplot as plt
import kernelregressionutils as kru
import networkx as nx
import argparse
from data import slurp, slurp_pickled_nx
import seaborn as sns; sns.set()
import torch as th
import networkx as nx

def poin2lor(embPoin):
    """Converts a set of poiints from Poincaré to Lorentz representation"""
    sqnorms = np.sum(embPoin ** 2, axis=1)[:, np.newaxis]
    embLor = np.hstack((1+sqnorms, 2*embPoin)) / (1-sqnorms)
    return embLor

def lorentzGeodesic(Y0, Y1):
    """Returns the set of geodesic distances between two set of points Y0 and Y1
    :param Y0: A set of P0 points in H_n seen as a set of row vectors in R^(n+1)
    :param Y1:  A set of P1 points in H_n seen as a set of row vectors in R^(n+1)
    :return:
    """
    return np.arccosh(-lorentzDotThresholded(Y0,Y1))

def lorentzDotThresholded(Y0, Y1):
    """Computes the Lorentz Riemannian product, thresholding values in (-1, 1)
    to the closest limit point
    :param Y0: A set of P0 points in H_n seen as a set of row vectors in R^(n+1)
    :param Y1: A set of P1 points in H_n seen as a set of row vectors in R^(n+1)
    :return: A matrix G of size P0 x P1 where each element (G)_ij = <(Y0)_i, (Y1)_j>_L
    """
    G = np.dot(Y0[:,1:], Y1[:,1:].T) - Y0[:,0][:,np.newaxis]*Y1[:,0][:,np.newaxis]
    G[ (G > 0) & (G < 1) ] =  1
    G[ (G >-1) & (G <=0) ] = -1
    return G

def lorentzDot(Y0, Y1):
    """Computes the Lorentz Riemannian product
    :param Y0: A set of P0 points in H_n seen as a set of row vectors in R^(n+1)
    :param Y1: A set of P1 points in H_n seen as a set of row vectors in R^(n+1)
    :return: A matrix G of size P0 x P1 where each element (G)_ij = <(Y0)_i, (Y1)_j>_L
    """
    return np.dot(Y0[:,1:], Y1[:,1:].T) - Y0[:,0][:,np.newaxis]*Y1[:,0][:,np.newaxis]

def lorentzSqNorm(y):
    """Returns the Lorentz squared norm evalueted with the Lorentz Riemannian product
    :param y: A spoint in H_n seen as a set of row vectors in R^(n+1)
    :return: the Lorentz norm of y
    """
    return lorentzDotThresholded(y,y)

def lorentzNorm(y):
    """Returns the Lorentz norm evaluated with the Lorentz Riemannian product
    :param y: A spoint in H_n seen as a set of row vectors in R^(n+1)
    :return: the Lorentz norm of y
    """
    return np.sqrt(lorentzDotThresholded(y,y))

def steepestAscent(Ytr, Y):
    """Computes the steepest ascent direction of the squared geodesic loss for the Lorentz hyperbolic model.
    h = g_L^{-1} grad f(Y)
    Note: H_n = Lorentz model of hyperbolic manifold
    :param Y: A point in H_n seen as a row vector in R^(n+1)
    :param Ytr: A set of P points in H_n seen as a set of row vector in R^(n+1)
    :return: A matrix (n+1) x P, where each column is the steepest ascent direction w.r.t to Y for the
    squared geodesic loss in H_n: d(Y, (Ytr)_i)^2
    """
    inners = lorentzDotThresholded(Ytr, Y) # col vector
    coefficients =  -2 * np.arccosh(-inners) / np.sqrt(inners**2 - 1)
    coefficients[inners == -1] = -2
    # step_control = lorentzGeodesic(Ytr, Y)**2
    step_control = 1
    return (step_control * coefficients * Ytr).T

def tangentSpaceProj(v, p):
    """Computes the projection of vector v in the space tangent to p
    :param v: A set of V vectors in the tangent space of p seen as row vectors in R^{n+1}
    :param p: A point in H_n seen as a row vector in R^{n+1}
    :return: A matrix V x (n+1) where each row is the steepest ascent direction of (v)_i"""
    p_tile = np.tile(p, (v.shape[0],1))
    return v - (lorentzDot(v, p)/lorentzDot(p,p)) * p_tile

def expMap(v, p):
    """Evaluates the exponential map of a vector v in the space tangent to p on the Lorentz model of
    hyperbolic manifold
    :param v: A row vector on the tangent space of p
    :param p: A point in H_n seen as a row vector in R^(n+1)
    :return: The point closest to p with geodesics having as an acceleration vector v
    """
    v_norm = np.sqrt(lorentzDot(v,v))
    if v_norm == 0:
        return p
    else:
        return np.cosh(v_norm) * p + np.sinh(v_norm) * (v / v_norm)

def gradLorentz(Ytr, Y):
    """Returns the Riemannian gradient of the squared geodesic distance for the Lorentz model of hyperbolic
    manifold of a point y in H_n with respect to a set of points Ytr in H_n.
    Squared geodesic distance in Lorentz model: d(y0,y1)^2 = arcosh( -<y0,y1>_L )^2
    <y0,y1>_L is the Lorentz Riemannian scalar product.

    :param Y: A point in H_n seen as a row vector in R^(n+1)
    :param Ytr: A set of P points in H_n seen as a set of row vector in R^(n+1)
    :return: A matrix (n+1),where each row i is the steepest gradient w.r.t the i-th point in Ytr
    """
    H = steepestAscent(Ytr, Y).T # row indexes training points, col indexes dim
    return tangentSpaceProj(H, Y)

def mAP(node, node_embed, graph, embed, embed_labels):
    neighs = [n for n in graph.neighbors(node)]
    D = lorentzGeodesic(embed, node_embed)
    embed_neighs =[embed_labels[idx] for idx in np.squeeze(np.argsort(D, axis=0)[:len(neighs)])] # labels of neighbours in the embedding

    accuracy = []
    for idx in range(1, len(neighs)+1):
        c = sum(neigh in embed_neighs[:idx] for neigh in neighs)
        accuracy.append(c/idx)

    return accuracy
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train euclidean on toy dataset')
    parser.add_argument('-dset', help='Dataset to embed', type=str, default='data/simple_supervised_embed_graph.p')
    parser.add_argument('-fset', help='Features dataset', type=str, default='data/toy_features.npy')
    parser.add_argument('-embed', help='Embedding', type=str, default='data/simple_supervised_embed.pth')
    parser.add_argument('-lr', help='Learning rate', type=float, default=0.003)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=50)
    parser.add_argument('-fout', help='Filename where to store model', type=str, default='data/dummy.out')
    parser.add_argument('-debug', help='Print debug output', action='store_true', default=False)
    opt = parser.parse_args()

    # Load graph from .tsv or .p, load
    if opt.dset[-4:] == '.tsv':
        #idx, objects = slurp(opt.dset)
        pass
    elif opt.dset[-2:] == '.p':
        graph = nx.read_gpickle(opt.dset)
        #idx, objects = slurp_pickled_nx(opt.dset, opt.f)
    print("Grapht type:", type(graph))
    adjacency = np.load(opt.fset)
    model = th.load(opt.embed) # dict_keys(['model', 'epoch', 'objects'])
    # edges = graph['edges']
    embedPoin = model['model']['lt.weight'].numpy()
    embed = poin2lor(embedPoin)
    # for idx in range(embed.shape[0]):
    #     print(idx)
    #     print(lorentzSqNorm(embed[idx][np.newaxis,:]), end="\n\n")
    print("Embedding matrix shape:", embed.shape)

    # Split training and test data
    testList = [101, 25, 35, 45, 50, 65, 80, 94, 100]
    #fathers: 16->101, 5->25, 7->35, 8->45, 9->50, 11->65, 13->80, 15->94, 16->100

    xtest =[]; ytest = []; ctest =[]
    xtrain =[]; ytrain = []; ctrain=[]
    for idx, embPoint in enumerate(embed):
        if idx in testList:
            xtest.append(adjacency[idx])
            ytest.append(embPoint)
            ctest.append(idx)
        else:
            xtrain.append(adjacency[idx])
            ytrain.append(embPoint)
            ctrain.append(idx)

    xtrain = np.array(xtrain)
    xtest  = np.array(xtest)
    ytrain = np.array(ytrain)
    ytest  = np.array(ytest)


    # #### DEBUG
    # xtrain = np.vstack((xtrain[0], xtrain[0]))
    # ytrain = np.vstack((ytrain[0], ytrain[0]))
    # xtest  = np.vstack((xtrain[0], xtrain[0]))
    # ytest = np.vstack((ytrain[0], ytrain[0]))

    valIdx = np.random.choice(xtrain.shape[0], int(xtrain.shape[0]*0.2))
    X = {'xtr':xtrain, 'xval':xtrain[valIdx]}
    Y = {'ytr':ytrain, 'yval':ytrain[valIdx]}


    loss = Loss('squaredLorentzGeodesic')


    #sigmaKRLS, lambdaKRLS = kru.KRLSvec_crossval(X=X, Y=Y)

    sigmaKRLS = 1.5
    lambdaKRLS = 1E-3

    Alphax = Alpha(kernel='gausskernel')
    losses = []
    ypred = []

    for idx, x in enumerate(xtest):
        x = x[np.newaxis, :]
        # Computes the estimator to be minimized
        alpha = Alphax.eval(X=xtrain, x=x, lam=lambdaKRLS, sigma=sigmaKRLS)
        print("Positive alphas: ", np.sum(alpha > 0), "/", alpha.size,
              "\t Average alpha magnitude:", np.mean(alpha),
              "\t Max alpha magnitude:", np.max(np.abs(alpha)))
        y_hist = np.full((opt.epochs, ytrain.shape[0]), np.inf)
        y0 = ytrain[np.random.randint(0, ytrain.shape[0])][np.newaxis,:]
        yt = y0
        eta = np.ones(opt.epochs) * opt.lr

        for iter in range(opt.epochs):
            pointwise_grad = gradLorentz(ytrain, yt)
            loss_grad =np.dot(alpha, pointwise_grad)[np.newaxis,:]
            yt = expMap(-eta[iter]*loss_grad, yt)
            if (iter % 100 == 0) or (iter == (opt.epochs-1)):
                print("Iteration %d, loss = %f" % (iter,loss.eval(ytest[idx][np.newaxis,:], yt)))
            if np.linalg.norm(loss_grad) < 1E-4:
                break
        losses.append(loss.eval(ytest[idx][np.newaxis,:], yt))
        ypred.append(yt)
    print("Mean average loss: %f" % (np.mean(losses)))

c = np.array(ctrain)

for idx, y in enumerate(ypred):
    print("Distance d(ypred,ytest): %f" % (lorentzGeodesic(y, ytest[idx][np.newaxis,:]) ), end='\n\n')
    # Computes distances to training set points
    D = lorentzGeodesic(np.array(ytrain), y)
    D_sort = np.argsort(D, axis=0)
    print("Nearest 10 neighbours of predicted node %d: " % (ctest[idx]))
    print(np.array(c[D_sort[:10]]).T)
    print("Distance to nearest neigh: %f" % (D[D_sort[0]]))
    # Computes distances of real point in embedding
    Dtrue = lorentzGeodesic(np.array(ytrain), ytest[idx][np.newaxis,:])
    Dtrue_sort = np.argsort(Dtrue, axis=0)
    print("Embedding Nearest 10 neighbours of node %d: " % (ctest[idx]))
    print(np.array(c[Dtrue_sort[:10]]).T)
    print("Distance to nearest neigh: %f" % (D[Dtrue_sort[0]]), end='\n\n\n\n')
    # acc = mAP(ctest[idx], y, graph, embed, ctrain)

