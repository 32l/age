from __future__ import division, print_function
import numpy as np

def compute_mean_ap(scores, labels):
    '''
    compute meanAP
    input:  scores    : M x N ndarray that contains N-dimension scores of M samples
            labels    : M x N ndarray that contains ground truth label (0 or 1)
    output: AP        : N x 1 array that contains AP for each class(attribute)
    output: meanAP    : mean(AP)
    '''
    assert(scores.shape == labels.shape)
    assert(scores.ndim == 2)
    M, N = scores.shape[0], scores.shape[1]

    # compute tp: column n in tp is the n-th class labels in descending order of the sample score.
    index = np.argsort(scores, axis = 0)[::-1, :]
    tp = labels.copy().astype(np.float)
    for i in xrange(N):
        tp[:, i] = tp[index[:,i], i]
    tp = tp.cumsum(axis = 0)

    m_grid, n_grid = np.meshgrid(range(M), range(N), indexing = 'ij')
    tp_add_fp = m_grid + 1    
    num_truths = np.sum(labels, axis = 0)
    # compute recall and precise
    rec = tp / num_truths
    prec = tp / tp_add_fp

    prec = np.append(np.zeros((1,N), dtype = np.float), prec, axis = 0)
    for i in xrange(M-1, -1, -1):
        prec[i, :] = np.max(prec[i:i+2, :], axis = 0)
    rec_1 = np.append(np.zeros((1,N), dtype = np.float), rec, axis = 0)
    rec_2 = np.append(rec, np.ones((1,N), dtype = np.float), axis = 0)
    AP = np.sum(prec * (rec_2 - rec_1), axis = 0)
    AP[np.isnan(AP)] = -1 # avoid error caused by classes that have no positive sample
    assert((AP <= 1).all())
    meanAP = AP[AP != -1].mean()
    return meanAP, AP


def compute_dist(P, Q):
    '''
    P and Q: n x d mat, each row is a sample's d-dim feature
    '''

    p2 = P.dot(P.T).diagonal().reshape(-1, 1)
    q2 = Q.dot(Q.T).diagonal().reshape(1, -1)
    pq = P.dot(Q.T)

    return p2 - 2*pq + q2
