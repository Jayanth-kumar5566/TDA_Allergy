import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas
import seaborn as sns

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
   #Not defined for l = total dimension
   ''' s is the matrix containing singular values in descending order
   l is the dimension we wish to reduce to not indices
   r is the orginal dimension not indices
   gamma is the number of rows or genes'''
   c=((gamma-l-1)*(r-l))/(gamma+r-(2*l))
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

#==================Healthy data=============
df=pandas.read_csv("./../../OD_Healthy 50 years and above (126 allergens).csv",index_col=0)
df=df.loc[:,df.isna().sum()==0]

b_df=pandas.read_csv("./../../OD_Bronchiectasis.csv",index_col=0)
b_df=b_df.loc[:,b_df.isna().sum()==0]


c_df=pandas.read_csv("./../../OD_COPD.csv",index_col=0)
c_df=c_df.loc[:,c_df.isna().sum()==0]

cols_tmp=set(df.columns).intersection(set(b_df.columns))

cols=set(c_df).intersection(cols_tmp)

df=df.loc[:,cols]
b_df=b_df.loc[:,cols]
c_df=c_df.loc[:,cols]

#=================Correlation matrix===========
'''
#They are correlated (both positively and negatively) - can do KNN

corrM = df.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(
    corrM,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=500),
    square=True, ax=ax
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.savefig("correlation.png",dpi=300)
'''

'''
#====Missing values imputation====
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights="uniform")
A=imputer.fit_transform(df)
'''

A=df.values


#Remove missing values

A[A<0]=0 #make negative values zero

A=A.transpose()

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

#Plotting Singular values after SVD decomposition for more insight into the PCA 
plt.plot(s,'-o')
plt.xlabel("Index")
plt.ylabel("Singular values")
plt.savefig("singular_values_patients.png",dpi=300)
plt.clf()

#Plotting a zoomed in-version
plt.plot(s[25:],'-o')
plt.xlabel("Index")
plt.ylabel("Singular values")
plt.savefig("singular_values_patients_zoomed.png",dpi=300)
plt.clf()



wold_score=[wold(s,i,r,gamma) for i in range(1,r)]
plt.plot(range(1,r),wold_score,'o-')
plt.ylabel("Wold Score")
plt.xlabel("PCA number of components")
plt.savefig("PCA.png",dpi=300)
#choose l based on the plot (the x-axis value)
l=int(input("Give in the value of l "))
N_new=N_hat(U,s,V,l)
N_b=basis(N_new,l)

#================Bronchieactasis Disease Data Creation==============

B=b_df.values
#Remove missing values
B[B<0]=0 #make negative values zero
B=B.transpose()

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

f_df=pandas.DataFrame(D_c,index=b_df.columns,columns=b_df.index).transpose()
f_df.to_csv("./../Results/bronchieactasis_data.csv")

del B,f_df,N_c,D_c
#==============COPD Disease Data Creation============

B=c_df.values
#Remove missing values
B[B<0]=0 #make negative values zero
B=B.transpose()

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

f_df=pandas.DataFrame(D_c,index=c_df.columns,columns=c_df.index).transpose()
f_df.to_csv("./../Results/COPD.csv")

