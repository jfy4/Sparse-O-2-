#!/anaconda/bin/python

import numpy as np
from math import exp
# np.set_printoptions(suppress=True, linewidth=200)
from itertools import product, combinations_with_replacement, imap
from scipy.special import iv
from time import time
from collections import Counter
import operator
from pprint import pprint
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def dict_sum(dict1, dict2, datatype):
    for key in dict2:
        try:
            dict1[key] += dict2[key]
        except KeyError:
            dict1[key] = dict2[key]
    return dict1

dictSumOp = MPI.Op.Create(dict_sum, commute=True)

prod = lambda factors: reduce(operator.mul, factors, 1)

list_add = lambda lol: list(sum(lol, []))

allcidl = lambda alist: list(set(list_add(alist)))

def tensorgen(D, beta, mu):
    """
    Creates a dictionary of tensor elements.

    Parameters
    ----------
    D : The absolute value of the largest Bessel function used.
    beta : The inverse temperature, or coupling.
    mu : The chemical potential.

    Returns
    -------
    dict : A dictionary with the nonzero elements of a tensor that would
           have been made using four indicies, now parameterized by three.

    """
    ttop = {}
    tbot = {}
    time0 = time()
    Is = np.sqrt([iv(a, beta) for a in range(-D, D+1)])
    It = np.sqrt([iv(a, beta)*exp(a*mu) for a in range(-D, D+1)])
    for t, x in product(range(-D, D+1), repeat=2):
        for tp in range(max([-D, t+x-D]), min([D, t+x+D])+1):
            ttop[(t, x, tp)] = np.array([[[[Is[t+D]*Is[tp+D]*It[x+D]*It[t+x-tp+D]]]]])
            tbot[(tp, t+x-tp, t)] = ttop[(t, x, tp)]
            # tbot[(tp, t+x-tp, t)] = np.array([[[[Is[t+D]*Is[tp+D]*It[x+D]*It[t+x-tp+D]]]]])
    time1 = time()
    if (rank == 0):
        print "tensor make time =", (time1-time0)
    return ttop, tbot

def gettop(ctdict, chrglist, D):
    """
    Creates the top half of the Q matrix.

    Parameters
    ----------
    ctdict : The dictionary corresponding to the T tensor in the original
             HOTRG formulation.
    chrglist : A list of the currect charge values being carried around for
               looping purposes, as well as histogram calculations.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary parameterizing the tensor which represents the top
           part of the Q matrix.

    """
    small = min(chrglist)
    large = max(chrglist)
    top = {}
    time0 = time()
    Dmin = ((rank*(2*D+1))/size)-D
    Dmax = (((rank+1)*(2*D+1))/size)-D
    for x in range(Dmin, Dmax):
        for tl, tr in product(chrglist, repeat=2):
            # print "x =", x
            for tp in range(max(small, tl+x-D, tr+x-D), min(large, tl+x+D, tr+x+D)+1):
                # print tl, tp, tl+x-tp
                t1s = ctdict[tl, x, tp].shape
                t2s = ctdict[tr, x, tp].shape
                temp1 = np.reshape(np.transpose(ctdict[tl, x, tp], (0, 3, 2, 1)), (t1s[0]*t1s[3], t1s[2]*t1s[1]))# need shape here
                temp2 = np.reshape(np.transpose(ctdict[tr, x, tp], (0, 3, 2, 1)), (t2s[0]*t2s[3], t2s[2]*t2s[1]))# need shape here
                try:
                    top[(tl, tr+x-tp, tr)] += np.transpose(np.reshape(np.dot(temp1, np.transpose(temp2)), (t1s[0], t1s[3], t2s[0], t2s[3])), (0, 2, 1, 3))# need shape here
                except KeyError:
                    top[(tl, tr+x-tp, tr)] = np.transpose(np.reshape(np.dot(temp1, np.transpose(temp2)), (t1s[0], t1s[3], t2s[0], t2s[3])), (0, 2, 1, 3))# need shape here
    the_result = comm.allreduce(top, op=dictSumOp)
    time1 = time()
    if (rank == 0):
        print "get top time =", (time1-time0)
    return the_result

def getbot(ctdict, chrglist, D):
    """
    Creates the bottom half of the Q matrix.

    Parameters
    ----------
    ctdict : The dictionary corresponding to the T tensor in the original
             HOTRG formulation.
    chrglist : A list of the currect charge values being carried around for
               looping purposes, as well as histogram calculations.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary parameterizing the tensor which represents the bottom
           part of the Q matrix.
           
    """
    bot = {}
    small = min(chrglist)
    large = max(chrglist)
    time0 = time()
    Dmin = ((rank*(2*D+1))/size)-D
    Dmax = (((rank+1)*(2*D+1))/size)-D
    for xp in range(Dmin, Dmax):
        for tl, tr in product(chrglist, repeat=2):
            for tp in range(max(small, -D+tl-xp, -D+tr-xp), min(large, D+tl-xp, D+tr-xp)+1):
                t1s = ctdict[tp, xp, tl].shape
                t2s = ctdict[tp, xp, tr].shape
                temp1 = np.reshape(np.transpose(ctdict[tp, xp, tl], (0, 2, 1, 3)), (t1s[0]*t1s[2], t1s[1]*t1s[3]))
                temp2 = np.reshape(np.transpose(ctdict[tp, xp, tr], (0, 2, 1, 3)), (t2s[0]*t2s[2], t2s[1]*t2s[3]))
                try:
                    bot[(tr, tp+xp-tr, tl)] += np.transpose(np.reshape(np.dot(temp1, np.transpose(temp2)), (t1s[0], t1s[2], t2s[0], t2s[2])), (0, 2, 1, 3))
                except KeyError:
                    bot[(tr, tp+xp-tr, tl)] = np.transpose(np.reshape(np.dot(temp1, np.transpose(temp2)), (t1s[0], t1s[2], t2s[0], t2s[2])), (0, 2, 1, 3))
    the_result = comm.allreduce(bot, op=dictSumOp)
    time1 = time()
    if (rank == 0):
        print "get bot time =", (time1-time0)
    return the_result

def getQ(top, bot, cvals, D):
    """
    Creates a block of the Q matrix provided by charge pair inputs.

    Parameters
    ----------
    top : A dictionary parameterizing the top part of the Q matrix.
    bot : A dictionary parameterizing the bottom part of the Q matrix.
    cvals : A list of tuples containing charge pairs that are relavent
            for building a block of the Q matrix.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary with the values of the block of the Q matrix.
    sizes : A list tuples containing the shape of the two leftmost legs
            of the Q matrix block.  Used to build the U tensor dictionary.

    """
    q = {}
    sizes = []
    time0 = time()
    for ttr, btr in cvals:
        for ttl, btl in cvals:
            rmin = max(-D, -D+ttr-ttl, -D+btl-btr)
            rmax = min(D, D+ttr-ttl, D+btl-btr)+1
            effDb = rmax-rmin
            Dmin = ((rank*(effDb))/size)+rmin
            Dmax = (((rank+1)*(effDb))/size)+rmin            
            for xp in range(Dmin, Dmax):
                t1s = top[ttl, xp, ttr].shape
                t2s = bot[btr, xp, btl].shape
                temp1 = np.reshape(top[ttl, xp, ttr], (t1s[0]*t1s[1], t1s[2]*t1s[3]))
                temp2 = np.reshape(bot[btr, xp, btl], (t2s[0]*t2s[1], t2s[2]*t2s[3]))
                temp3 = np.dot(temp1, np.transpose(temp2))
                temp3 = np.transpose(np.reshape(temp3, (t1s[0], t1s[1], t2s[0], t2s[1])), (0,2,1,3))
                try:
                    q[ttr, btr, ttl] += np.reshape(temp3, (t1s[0]*t2s[0], t1s[1]*t2s[1]))
                except KeyError:
                    q[ttr, btr, ttl] = np.reshape(temp3, (t1s[0]*t2s[0], t1s[1]*t2s[1]))
        sizes.append((t1s[1], t2s[1]))
    the_result = comm.allreduce(q, op=dictSumOp)
    time1 = time()
    return the_result, sizes

def blockeev(qblock, cvals):
    """
    Calculates the eigenvalues and eigenvectors of a Q matrix block.

    Parameters
    ----------
    qblock : The dictionary containing the values of the block from the
             Q matrix.
    cvals : A list of tuples containing the pairs of charges associated with
            the leftmost legs of the Q matrix block.

    Returns
    -------
    evals : An array of the eigenvalues of the block.
    evecs : An array of the eigenvectors of the block.

    """
    time0 = time()
    for i, j in cvals:
        temp = []
        for k in zip(*cvals)[0]:
            temp.append(qblock[i, j, k])
        temp = np.vstack(temp)
        if ((i, j) == cvals[0]):
            Qmat = temp
        else:
            Qmat = np.hstack((Qmat, temp))
    a, b = np.linalg.eigh(Qmat)
    time1 = time()
    return a, b

def chrgsets(chrglist):
    """
    Takes a list of charge values and returns all possible charge
    combinations in order, and which indices correspond to those
    charges, also in order.
    
    Parameters
    ----------
    chrglist : A list of charge values, possibly degenerate.
    
    Returns
    -------
    charge_range   : An array of all sums of all possible pairs of charge values
                     one can make with the given list.
    charge_indices : An array of lists whos elements are tuples which
                     gives the pair of charges associated with each
                     tensor leg.

    """
    ls = len(chrglist)
    idxmaster = []
    time0 = time()
    charges = sorted(list(set(imap(sum, combinations_with_replacement(chrglist, 2)))))
    for charge in charges:
        idxlist = []
        for i,j in product(chrglist, repeat=2):
            if (i+j == charge):
                idxlist.append((i,j))
            else:
                pass
        idxmaster.append(idxlist)
    time1 = time()
    if (rank == 0):
        print "charge build time =", (time1-time0)
    return (charges, idxmaster)

def cass(clist, vlist, idxlist, slist):
    """
    Makes a dictionary associating a charge value to vectors, charge pairs,
    and shapes.

    Parameters
    ----------
    clist : A list of charge values, possibly degenerate.
    vlist : A list of vectors.  Some vectors can share a common charge.
    idxlist : A list of charge-pair tuples. Each charge-pair tuple is associated
              with a tensor shape tuple.
    slist : A list of tuples representing the shape of the leftmost legs of
            a block from the Q matrix.  Each shape tuple is associated with
            a charge pair.

    Returns
    -------
    dict : A dictionary containing all of the vectors, charge-pairs, and shape-pairs
           associated with a charge value (block).

    """
    vecs = {}
    time0 = time()
    for c, v, i, s in zip(clist, vlist, idxlist, slist):
        try:
            vecs[c][0] = np.vstack((vecs[c][0], v))
            # temp = vecs[c][0].shape
            # vecs[c][0] = np.reshape(vecs[c][0], temp)
        except KeyError:
            vecs[c] = [v, i, s]
            temp = len(vecs[c][0])
            vecs[c][0] = np.reshape(vecs[c][0], (-1, temp))
    time1 = time()
    if (rank == 0):
        print "get cass time =", (time1-time0)
    return vecs

def getU(vecs):
    """
    Creates a dictionary containing the non-zero elements of the U tensor
    used for updating.

    Parameters
    ----------
    vecs : A dictionary which associates the necessary vectors, charge charge-pairs,
           and shape-pairs with a charge used to build the U tensor.

    Returns
    -------
    dict : A dictionary with the non-zero elements of the U tensor used for updating.
           It is parameterized with two charge values which fix the output leg.

    Notes
    -----
    If this is correct the output order for the sub-indices is
    (top, bot, alpha).

    """
    cvdict = {}
    time0 = time()
    for c in vecs:
        s = 0
        e = 0
        for i, j in zip(vecs[c][1], vecs[c][2]):
            e += prod(j)
            xx = (vecs[c][0][:,s:e]).shape
            # xx = np.transpose(vecs[c][0][:,s:e]).shape
            #print xx
            #print (j + (xx[1],))
            # cvdict[i] = np.reshape(np.transpose(vecs[c][0][:,s:e]), (j + (xx[1],)))
            cvdict[i] = np.reshape(np.transpose(vecs[c][0][:,s:e]), (j + (xx[0],)))
            s = e
    time1 = time()
    if (rank == 0):
        print "make U time =", (time1-time0)
    return cvdict

def update(ttop, tbot, udict, cidladded, D):
    """
    Creates a dictionary with the non-zero elements of an updated tensor after
    having gone through an iteration of HOTRG.

    Parameters
    ----------
    ctdict : A dictionary parameterizing the T tensor.
    udict : A dictionary parameterizing the U tensor.
    cidladded : A list of charge-pairs used for updating.  These are the reduced
                selected charge pairs which are used by udict.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary parameterizing the updated T tensor.  It is parameterized
           using three charge values again which imply the fourth.

    """
    topret = {}
    botret = {}
    time0 = time()
    chrgsize = len(cidladded)
    for tl, bl in cidladded:
        for tr, br in cidladded:
            rmin = max(-D, -D+tl-tr, -D+br-bl)
            rmax = min(D, D+tl-tr, D+br-bl)+1
            effDb = rmax-rmin
            Dmin = ((rank*(effDb))/size)+rmin
            Dmax = (((rank+1)*(effDb))/size)+rmin
            for k in range(Dmin, Dmax):
                uls = udict[tl, bl].shape
                tts = tbot[tr, k, tl].shape
                # print tr, k, tl, tts
                top = np.dot(np.reshape(np.transpose(udict[tl, bl], (2,1,0)), (uls[2]*uls[1], uls[0])),
                             np.reshape(tbot[tr, k, tl], (tts[0], tts[1]*tts[2]*tts[3])))
                top = np.reshape(np.transpose(np.reshape(top, (uls[2], uls[1], tts[1], tts[2], tts[3])),
                                              (0, 3, 2, 1, 4)),
                                 (uls[2]*tts[2], tts[1]*uls[1]*tts[3]))

                urs = udict[tr, br].shape
                tbs = ttop[bl, k, br].shape
                bot = np.dot(np.reshape(np.transpose(ttop[bl, k, br], (0,3,2,1)), (tbs[0]*tbs[3]*tbs[2], tbs[1])),
                             np.reshape(np.transpose(udict[tr, br], (1,0,2)), (urs[1], urs[0]*urs[2])))
                bot = np.reshape(np.transpose(np.reshape(bot, (tbs[0], tbs[3], tbs[2], urs[0], urs[2])),
                                              (3, 0, 2, 1, 4)),
                                 (urs[0]*tbs[0]*tbs[2], tbs[3]*urs[2]))
                try:
                    topret[(tl+bl, k+tr-tl, tr+br)] += np.transpose(np.reshape(np.dot(top, bot), (uls[2], tts[2], tbs[3], urs[2])),
                                                                       (0, 3, 1, 2))
                    botret[(tr+br, bl+k-br, tl+bl)] = topret[(tl+bl, k+tr-tl, tr+br)]
                except KeyError:
                    topret[(tl+bl, k+tr-tl, tr+br)] = np.transpose(np.reshape(np.dot(top, bot), (uls[2], tts[2], tbs[3], urs[2])),
                                                                      (0, 3, 1, 2))
                    botret[(tr+br, bl+k-br, tl+bl)] = topret[(tl+bl, k+tr-tl, tr+br)]
    time1 = time()
    if (rank == 0):
        print "update tensor time =", (time1-time0)
    top_result = comm.allreduce(topret, op=dictSumOp)
    bot_result = comm.allreduce(botret, op=dictSumOp)
    return top_result, bot_result

def tensor_norm(ctdict, chrglist, D):
    norm = 0.0
    for i, j in product(chrglist, repeat=2):
        for k in range(max(-D, -D+j-i), min(D, D+j-i)+1):
            norm += np.einsum('abcd, abcd', ctdict[i,k,j], ctdict[i,k,j])
    norm = np.sqrt(norm)
    return norm

def normalize(d, norm):
    for k, v in d.iteritems():
        d[k] = v/norm

def tensor_trace(ctdict, chrglist, D):
    trace = 0.0
    for i in chrglist:
        for k in range(-D, D+1):
            trace += np.einsum('aa', np.einsum('aaij', ctdict[i,k,i]))
    return trace

def dict_transpose(thedict, chrglist, D):
    newdict = {}
    for tl, tr in product(chrglist, repeat=2):
        for x in range(max(-D, -D+tr-tl), min(D, D+tr-tl)+1):
            newdict[(tr, tl+x-tr, tl)] = thedict[(tl, tx, tr)]
    return newdict

def getlists(O, Top, Bot, D):
    """
    Returns the good values of charge, vectors, charge pairs, and pair shapes.

    Paramters
    ---------
    O : A tuple of lists of charges and charge pairs.
    Top : The top part of the Q matrix.
    Bot : The bottom part of the Q matrix.
    D : The absolute value of the largest Bessel used.

    Returns
    -------
    clist : A list of the charges that made it through the slection.
    vlist : A list of vectors that made is through the selection.
    idxlist : The pairs of charges that match with the vectors and charges.
    slist : A list of the sizes of the charge pairs.

    """
    Dbond = 2*D+1
    clist = list(np.zeros((Dbond)))  # initialize lists to be filled                                                
    elist = list(np.zeros((Dbond)))
    vlist = list(np.zeros((Dbond)))
    idxlist = list(np.full((Dbond), 0.1))
    slist = list(np.zeros((Dbond)))
    globaltime0 = time()
    for c, l in O:              # loop through charges and pairs                                                    
        Q, sizes = getQ(Top, Bot, l, D) # make Q and the pair sizes                                             
        e, v = blockeev(Q, l)           # find the eigenvalues and vectors                                      
        idx = e.argsort()
        e = e[idx]
        v = v[:, idx].T
        for i in range(len(e)):
            if (e[i] >= min(elist)): # fill the lists with the best e and v                                         
                ii = np.argmin(elist)
                elist.pop(ii)
                elist.append(e[i])
                idxlist.pop(ii)
                idxlist.append(l)
                clist.pop(ii)
                clist.append(c)
                vlist.pop(ii)
                vlist.append(v[i])
                slist.pop(ii)
                slist.append(sizes)
            else:
                pass
    globaltime1 = time()
    if (rank == 0):
        print "total Q and eigs time =", (globaltime1-globaltime0)
    return clist, vlist, idxlist, slist
