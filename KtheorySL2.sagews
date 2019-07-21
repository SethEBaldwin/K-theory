import itertools
import numpy as np
import math

### INSTRUCTIONS:
### d(k,n,m) computes the structure constant d^k_{n,m}
###     for the basis of structure sheaves of opposite Schubert varities
### b(k,n,m) computes the structure constant b^k_{n,m}
###     for the dual basis of ideal sheaves of boundaries of opposite Schubert varities
### c(k,n,m) computes the structure constant c^k_{n,m}
###     for the Schubert basis in cohomology
### In K-theory, x and y represent e^{-\alpha_0} and e^{-\alpha_1} respectively
### Note that sign-alternation in K-theory holds in the variables 
###     e^{-\alpha_0} - 1 and e^{-\alpha_1} - 1
### In cohomology, x and y represented \alpha_0 and \alpha_1 respectively

R.<x,y> = LaurentPolynomialRing(QQ)

### K-THEORY

d_dict = {}

def strictly_inc_seqs(length,start,stop):
    return list(itertools.combinations(range(start,stop+1),length))

def inc_seqs(length,start,stop):
    return list(itertools.combinations_with_replacement(range(start,stop+1), length))

def e(k,m):
    if m % 2 == 0: return (-1)**(k+m+1) * x**((m/2)**2) * y**((m/2)**2+(m/2))
    else: return (-1)**(k+m+1) * x**((((m-1)/2)+1)**2) * y**(((m-1)/2)**2+(m-1)/2)

def beta(l,m):
    if m % 2 == 0: return x**(l-1) * y**(l)
    else: return x**(l) * y**(l-1)

# takes a sequence zs of increasing indices between 1 and m and returns True if the 
# corresponding subword of w_m equals w_n under the star operation, False otherwise
def reduce_star_equals(zs,n,m): 
    zs01 = [ ((z - m) % 2) for z in zs ]
    reduced_zs01 = [zs01[0]] + [ zs01[i] for i in range(1,len(zs)) if zs01[i] != zs01[i-1] ]
    return (len(reduced_zs01) == n and reduced_zs01[-1] == 0)

# returns a list of increasing sequences of indices from 1 to m such that the corresponding 
# subword of the reduced expression of w_m equals w_n under the star operation
def star_subsequences(n,m):
    seqs = []
    for i in range(n,m+1):
        for seq in strictly_inc_seqs(i,1,m):
            seqs.append(seq)
    return [ seq for seq in seqs if reduce_star_equals(seq,n,m) ]

def d_loc(n,m):
    return sum([ (-1)**n*np.product([ (beta(l,m)-1) for l in seq ]) for seq in star_subsequences(n,m) ])

def chi(seq, k):
    return x**(sum([ seq[i] for i in range(len(seq)) if (i+1) % 2 == k % 2 ])) * y**(sum([ seq[i] for i in range(len(seq)) if (i+1) % 2 != k % 2]))

def d(k,n,m):
    if d_dict.get((k,n,m)): return d_dict.get((k,n,m))
    else:
        if n == 1:
            if k < m: return 0
            elif k == m: return 1 + e(k,m)
            else: dknm = e(k,m)*(sum([ chi(seq,k) for seq in inc_seqs(k-m,1,m) ]) + sum([ chi(seq,k-1) for seq in inc_seqs(k-1-m,1,m) ]))
        else:
            if k < max(n,m): return 0
            elif k == max(n,m): dknm = d_loc(min(n,m),max(n,m))
            else: dknm = (1/(d(n,1,n) - d(k,1,k)))*(sum([ d(i,n,m)*d(k,1,i) for i in range(max(n,m),k) ]) - sum([ d(j,1,n)*d(k,j,m) for j in range(n + 1, k + 1) ]))
        d_dict[(k,n,m)] = dknm
        return dknm

def b(k,n,m):
    return sum([ d(l,n,m) - d(l,n+1,m) - d(l,n,m+1) + d(l,n+1,m+1) for l in range(k+1) ])

### COHOMOLOGY

c_dict = {}

def q(m):
    return int(math.ceil(m/2)**2)*x + int((math.floor(m/2)**2 + math.floor(m/2)))*y

def Q(d,i,j):
    degs = WeightedIntegerVectors(d, [1]*(j+1))
    return sum([ np.product([ q(i+l)**d[l] for l in range(j+1) ]) for d in degs ])

def c(k,n,m):
    if c_dict.get((k,n,m)): return c_dict.get((k,n,m))
    else:
        if n > m: return c(k,m,n)
        elif n == 1:
            if k == m: return q(m)
            elif k == m+1: return m+1
            else: return 0
        elif k < m or k > n+m: return 0
        else:
            cknm = (math.factorial(k)*Q(n+m-k,m,k-m)/math.factorial(m) - sum([ math.factorial(l)*Q(n-l,1,l-1)*c(k,l,m) for l in range(max(k-m,1), min(n-1,k)+1) ]))/math.factorial(n)
            c_dict[(k,n,m)] = cknm
            return cknm

### EXAMPLE CALCULATIONS:

X = var('X')
Y = var('Y')

# X and Y denote the variables e^{-\alpha_0} - 1 and e^{-\alpha_1} - 1 respectively
print(d(12,1,5).substitute(x=X+1,y=Y+1).factor())
print('\n')
print(d(7,3,7).substitute(x=X+1,y=Y+1).factor())
print('\n')
print(d(18,2,3).substitute(x=X+1,y=Y+1).factor())
print('\n')
print(b(4,2,3).substitute(x=X+1,y=Y+1).factor())
print('\n')
print(c(32,27,15).factor())
