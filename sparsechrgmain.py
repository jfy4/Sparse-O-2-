#!/anaconda/bin/python

import chrgsectorfuncs as csf
import numpy as np
from time import time
from sys import argv


D = int(argv[1])
Dbond = 2*D+1
beta = np.float64(argv[2])
mu = np.float64(argv[3])
nit = int(argv[4])

if (csf.rank == 0):
    print "------------------------------------"
    print "D =", D
    print "Dbond =", Dbond
    print "Beta =", beta
    print "Mu =", mu
    print "iterations =", nit
    print "------------------------------------"
# stdout.flush()

initchrg = range(-D, D+1)
T, B = csf.tensorgen(D, beta, mu)
if (csf.rank == 0):
    print "partition function =", csf.tensor_trace(T, initchrg, D)
norm = csf.tensor_norm(T, initchrg, D)
if (csf.rank == 0):
    print norm
csf.normalize(T, norm)
csf.normalize(B, norm)
timelist = []

for i in range(nit):
    if (csf.rank == 0):
        print "------------------------------------"
        print "iteration", i+1
        print "------------------------------------"
    charges, cidl = csf.chrgsets(initchrg)
    #print charges
    O = zip(charges, cidl)
    Top = csf.gettop(T, initchrg, D)
    # pprint(Top)
    Bot = csf.getbot(B, initchrg, D)
    # print T
    clist = list(np.zeros((Dbond)))
    elist = list(np.zeros((Dbond)))
    vlist = list(np.zeros((Dbond)))
    idxlist = list(np.full((Dbond), 0.1))
    slist = list(np.zeros((Dbond)))
    # rlist = {}
    globaltime0 = time()
    for c, l in O:
        # print c, l
        Q, sizes = csf.getQ(Top, Bot, l, D)
        # print "end block"
        e, v = csf.blockeev(Q, l)
        idx = e.argsort()
        e = e[idx]
        v = v[:, idx].T
        # print sizes
        # print block
        for i in range(len(e)):
            if (e[i] >= min(elist)):
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
    if (csf.rank == 0):
        print "total Q and eigs time =", (globaltime1-globaltime0)
    # pprint(idxlist)
    # pprint(slist)
    # rlist = Counter(clist)
    # print cidl
    tempcass = csf.cass(clist, vlist, idxlist, slist)
    # pprint(tempcass)
    U = csf.getU(tempcass)
    # pprint(U)
    cidladded = csf.allcidl(idxlist)
    # pprint([(k, U[k], U[k].shape) for k in cidladded])
    lt1 = time()
    T, B = csf.update(T, B, U, cidladded, D)
    lt2 = time()
    if (csf.rank == 0):
        timelist.append(lt2-lt1)
    initchrg = list(set(clist))
    norm = csf.tensor_norm(T, initchrg, D)
    if (csf.rank == 0):
        print norm
        print "partition function =", csf.tensor_trace(T, initchrg, D)
    csf.normalize(T, norm)
    csf.normalize(B, norm)
    # pprint(T)
    # stdout.flush()
if (csf.rank == 0):
    np.save("./timelist" + str(D) + ".npy", timelist)
del(T, B, U, Top, Bot)





