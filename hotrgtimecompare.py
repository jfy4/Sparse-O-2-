#!/anaconda/bin/python

from o2funcs import *
from sys import argv

D = int(argv[1])
beta = np.float64(argv[2])
mu = np.float64(argv[3])
nit = int(argv[4])
Dbond = 2*D+1

print "--------------------------"
print "Dbond =", Dbond
print "Beta =", beta
print "Mu =", mu
print "number of iterations =", nit
print "--------------------------"

T,_ = tensorgen(D, beta, mu)
norm = np.linalg.norm(T)
print "pf =", np.einsum('aa', np.einsum('ijaa', T))
T = T/norm
timelist = []
for i in range(nit):
    print "---------------"
    print "iteration ", i+1
    print "---------------"
    U = getU(T, Dbond)
    lt0 = time.time()
    T = update(T, U)
    lt1 = time.time()
    timelist.append(lt1-lt0)
    norm = np.linalg.norm(T)
    print norm
    print "pf =", np.einsum('aa', np.einsum('ijaa', T))
    T = T/norm
del(T, U)
# np.save("./timelisthotrg" + str(D) + ".npy", timelist)
