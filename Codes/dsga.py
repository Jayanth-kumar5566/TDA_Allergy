import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import linear_model

#----------FLAT Reconstruction---------------
def drop(A,ind):
    ''' A is a numpy matrix 
        ind is the index of the column to be removed'''
    p=A[:,ind]
    A_1=A[:,0:ind]
    A_2=A[:,ind+1:A.shape[1]]
    A_new=numpy.hstack((A_1,A_2))
    return (p,A_new)

def flat(A,ind):
    B=drop(A,ind)
    X=B[1]
    y=B[0]
    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(X,y)
    #print X.shape
    #print reg.coef_.shape
    return numpy.dot(X,reg.coef_)
#----------------Dimension Reduction------------------
#chosing appropriate dimension l

def wold(s,l,r,gamma):
   ''' s is the matrix containing singular values in descending order
   l is the dimension we wish to reduce to not indices
   r is the orginal dimension not indices
   gamma is the number of rows or genes'''
   c=(gamma-l-1)*(r-l)/(gamma+r-2*l)
   l=l-1 #makes it as index
   r=r-1 #makes it as index
   w_1=(s[l])**2/numpy.sum(numpy.square(s[l+1:]))
   return w_1*c


def N_hat(U,s,V,l):
    ''' returns the dimension reduced array '''
    s[l:]=0
    S=numpy.diag(s)
    N=numpy.matmul(U,numpy.matmul(S,V))
    return N

#rank of N_hat is reduced to l
 #========Finding the basis of the matrix

def basis(N,l,method='QR'):
    ''' The numpy matrix returned after N_hat'''
    if method != 'QR':
        return scipy.linalg.orth(N)
    else:
        q,r=numpy.linalg.qr(N)
        return q[:,:l]

def project(A,v):
    ''' A is a numpy array of basis of the subspace, output of basis
    v is the column of A(orginal matrix)
    Vn is the projected data on N_l space
    and V_np is the projected data perpendicular to it'''
    P_2=numpy.linalg.inv(numpy.matmul(A.transpose(),A))
    P=numpy.matmul(numpy.matmul(A,P_2),A.transpose())
    Vn=numpy.dot(P,v)
    V_np=v-Vn
    return (Vn,V_np)
#======================================================

#----Normal Data set creation----------------------
A=numpy.random.random((10,10))*100 #Normal people
#--------------------------------------------
dim=A.shape
r=dim[1]
gamma=dim[0]
N=[]
for i in range(r):
    N_i=flat(A,i)
    N.append(N_i)
N=numpy.stack(N,axis=1)
#Take the value of l such that wold spikes up
#note l is the dimension not the index
U, s, V = numpy.linalg.svd(N, full_matrices=False)
for i in range(1,r):
    plt.scatter(i,wold(s,i,r,gamma))
plt.show()
#choose l based on the plot (the x-axis value)
l=int(raw_input("Give in the value of l "))
N_new=N_hat(U,s,V,l)
N_b=basis(N_new,l)

#================Disease Data Creation==============
B=numpy.random.random((10,10))*100 #Diseased people
(p,q)=B.shape

N_c=[]
D_c=[]
for i in range(q):
    p,B_new=drop(B,i)
    (V_n,V_np)=project(N_b,p)
    N_c.append(V_n)
    D_c.append(V_np)
N_c=numpy.stack(N_c,axis=1)
D_c=numpy.stack(D_c,axis=1)
