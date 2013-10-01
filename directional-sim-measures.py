import sys
import numpy as np
import math

def check(u,v):
    if not (len(u) == len(v)):
        sys.exit("Fatal error: input vectors not equal in length.")
    if not (u.dtype == float and v.dtype == float):
        sys.exit("Fatal error: input vectors not of float type.")

def findIntersectedFeatures(u,v):
    common = []
    for i in range(len(u)):
        if not (u[i] == 0) and not (v[i] == 0):
            common.append(i)
    indices = np.array(common)
    inters_u = u[indices]
    inters_v = v[indices]
    return (inters_u, inters_v)

def vectorCosine(u,v):
    check(u,v)
    dot = np.dot(u,v)
    u_modulus = np.math.sqrt((u*u).sum())
    v_modulus = np.math.sqrt((v*v).sum())
    cos_angle = dot / (u_modulus * v_modulus) # cosine of angle between u and v
    return cos_angle

def lin(u,v):
    check(u,v)
    (inters_u, inters_v) = findIntersectedFeatures(u,v)
    tmp = inters_u + inters_v
    result = tmp.sum() / (u.sum() + v.sum())
    return result

def alphaSkew(u,v,alpha=.99):
    check(u,v)
    result = 0
    oneminusalpha = 1 - alpha
    for i in range(len(u)):
        f_u = u[i]
        f_v = v[i]
        if not f_v == 0: # if f_v is zero, everything goes to zero (to avoid error caused by log(0))
            tmp1 = math.log(f_v / ( alpha * f_u + oneminusalpha * f_v ) ) * f_v
            result += tmp1
    return result

def WeedsPrec(u,v):
    check(u,v)
    (inters_u, inters_v) = findIntersectedFeatures(u,v)
    result = inters_u.sum() / u.sum()
    return result

def balPrec(u,v):
    check(u,v)
    lin1 = lin(u,v)
    wPrec1 = WeedsPrec(u,v)
    result = math.sqrt(lin1*wPrec1)
    return result

def ClarkeDE(u,v):
    check(u,v)
    (inters_u, inters_v) = findIntersectedFeatures(u,v)
    numerator = 0
    for i in range(len(inters_u)):
        value = min(inters_u[i], inters_v[i])
        numerator += value
    result = numerator / u.sum()
    return result

def findIntersectedFeatures(u,v):
    common = []
    for i in range(len(u)):
        if not (u[i] == 0) and not (v[i] == 0):
            common.append(i)
    indices = np.array(common)
    inters_u = u[indices]
    inters_v = v[indices]
    return (inters_u, inters_v)

# AP requires ranks. What to do with ties? (cf. zeroes)
def APinc(u,v):
    check(u,v)
    rankVector = rankdata(u)
    for i in range(len(u)):
        pass
        #rank = i+1 # technically ranks range from 1 to n, but I won't bother here
        #        (includedFeatsAtRank, dummy) = findIntersectedFeatures(u[:i],v[:i])
        #precAtRank = len(includedFeatsAtRank) / len(u[:i])
        #precAtRankPrevious=precAtRank

def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__, reverse=True)

def rankdata(a):
    """
    Gives a rank vector, with averages for ties.
    """
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in xrange(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in xrange(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

data = np.genfromtxt('test', delimiter=' ')
print(data)
u = data[0]; v = data[1]
#u = np.array([1,0,3],dtype=float)
#v = np.array([1,2,0],dtype=float)
# print "features: should be"
# print "(array([ 1.,  2.]), array([ 1.,  1.]))"
# print "and is:"
# print findIntersectedFeatures(u,v)
# print "cosine: 0.707106781187"
# print vectorCosine(u,v)
# print "lin: 0.714285714286"
# print lin(u,v)
# print "alpha-skew: 3.9170355472516905"
# print alphaSkew(u,v)
# print "weedsprec: 0.75"
# print WeedsPrec(u,v)
# print "balPrec: 0.731925054711"
# print balPrec(u,v)
# print "ClarkeDE: 0.5"
# print ClarkeDE(u,v)
# print "balAPinc: still missing"
