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

runchrglist = []                # an empty list to fill with the running charges
initchrg = range(-D, D+1)       # the initial running charges
runchrglist.append(initchrg)    # append the running charges
totalgtime0 = time()             # begin total timing
T, B = csf.tensorgen(D, beta, mu) # generate the initial tensors
if (csf.rank == 0):
    print "partition function =", csf.tensor_trace(T, initchrg, D)
norm = csf.tensor_norm(T, initchrg, D)
if (csf.rank == 0):
    print norm
csf.normalize(T, norm)          # normalize the tensors
csf.normalize(B, norm)
timelist = []

for i in range(nit):
    if (csf.rank == 0):
        print "------------------------------------"
        print "iteration", i+1
        print "------------------------------------"
    charges, cidl = csf.chrgsets(initchrg) # generate total charges and pairs
    O = zip(charges, cidl)
    Top = csf.gettop(T, initchrg, D) # make the top of Q
    Bot = csf.getbot(B, initchrg, D) # make the bottom of Q
    clist, vlist, idxlist, slist = csf.getlists(O, Top, Bot, D)
    tempcass = csf.cass(clist, vlist, idxlist, slist) # puts the charges, vectors, pairs, and shapes together
    U = csf.getU(tempcass)                            # gets the U for updating
    cidladded = csf.allcidl(idxlist)                  # makes a list of unique pairs
    lt1 = time()
    T, B = csf.update(T, B, U, cidladded, D) # make a new updated tensor
    lt2 = time()
    if (csf.rank == 0):
        timelist.append(lt2-lt1)
    initchrg = list(set(clist)) # makes a new list of running charges
    runchrglist.append(clist) # appends it to the list of running charges
    norm = csf.tensor_norm(T, initchrg, D)
    if (csf.rank == 0):
        print norm
        print "partition function =", csf.tensor_trace(T, initchrg, D)
    csf.normalize(T, norm)
    csf.normalize(B, norm)

totalgtime1 = time()            # end total timing
if (csf.rank == 0):
    print "total iteration time =", (totalgtime1-totalgtime0)
    # np.save("./timelist" + str(D) + ".npy", timelist)
    np.save("./chrghistdata_D" + argv[1] + "_b" + argv[2] + "_m" + argv[3] + "_L" + argv[4] + ".npy",
            np.asarray(runchrglist))
del(T, B, U, Top, Bot)





