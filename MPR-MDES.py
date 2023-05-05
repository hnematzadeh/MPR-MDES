import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, StandardScaler
from sklearn import tree,svm
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from collections import Counter, defaultdict
import random
import math
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from time import time
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
from sklearn import metrics 



# Load the datasets at your local drive
# colon; labels are 0 or 1
df = pd.read_excel ('D:\Colon.xlsx', header=None)
df.iloc[:,df.shape[1]-1].replace({'Normal':1, 'Tumor':2},inplace=True)

# CNS; labels are 1 or 2
df = pd.read_excel ('D:\CNS.xlsx', header=None)
df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# Leukemia-2c; labels are 1 or 2
df = pd.read_excel ('D:\Leukemia.xlsx', header=None)

#SMK
df = pd.read_csv ('D:\SMK.csv', header=None)
df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# GLI
df = pd.read_csv ('D:\GLI.csv', header=None)
df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# covid-3c
df = pd.read_csv ('D:\Covid.csv', header=None)
df.iloc[:,df.shape[1]-1].replace({'no virus':1, 'other virus':2, 'SC2':3},inplace=True)

#Leukemia-3c
df = pd.read_excel ('D:\Leukemia_3c.xlsx', header=None)

#MLL-3c
df = pd.read_excel ('D:\MLL.xlsx', header=None)

#SRBCT-4c
df = pd.read_excel ('D:\SRBCT.xlsx', header=None)



X=df.iloc[:,0:df.shape[1]-1]
# X=pd.DataFrame(scale(X))
y=df.iloc[:,df.shape[1]-1]

##### Calculating  quantity of each label in a
labels=np.unique(y)
a = {}

c=1
for i in range (len(labels)):
    # dynamically create key
    key = c
    # calculate value
    value = sum(y==labels[i])
    a[key] = value 
    
    c +=1
#################Pattern recognition feature selection with  O (mnlogn) time complexity###################
t0=time()
from operator import itemgetter
import itertools
alpha=np.zeros(df.shape[1]-1)
sorted_alpha=np.zeros(df.shape[1]-1)
for w in range(df.shape[1]-1):
    print(w) 
    newdf=df.sort_values(w)   ##  O(n^2)
    z = [(x[0], len(list(x[1]))) for x in itertools.groupby(newdf[newdf.shape[1]-1])] ## O(n)
    d=1
    for i in range (len(labels)): ##  O(n)
         d= d * (max(filter(lambda f: f[0] == labels[i], z), key=itemgetter(1))[1] / a[i+1])
    alpha[w]=np.sqrt(d)    
t1=time()
print(t1-t0)         
        
    ####################construct MC df########################################  
limit=20
zz=(alpha).argsort()[::-1]
mat1=np.zeros((df.shape[0],limit+1))
#mat1=np.zeros((df.shape[0],201))
for i in range(limit):
#
   mat1[:,i]=X.iloc[:,zz[i]]
#mat1[:,200]=y
mat1[:,limit]=y


Xn=mat1[:,0:mat1.shape[1]-1]

Xn=pd.DataFrame(Xn)
yn=mat1[:,limit]
################################################  


################### Fitness function######################################
def Fitness(Xnew_n,ynew_n):
   from sklearn.metrics import confusion_matrix, classification_report
   from sklearn import tree,svm

   s=np.zeros(10)
   Precision=np.zeros(10)
   Recall=np.zeros(10)
   F1_score=np.zeros(10)
   for i in range(10):
      X_train, X_test, y_train, y_test = train_test_split(Xnew_n,ynew_n, test_size=0.2, stratify=y)
      dectree = tree.DecisionTreeClassifier()
      dectree.fit(X_train,y_train)
      s[i]=dectree.score(X_test,y_test)
      cm=confusion_matrix(y_test,dectree.predict(X_test))
      Precision[i] = metrics.precision_score(y_test, dectree.predict(X_test), average='micro')
      Recall[i] = metrics.recall_score(y_test, dectree.predict(X_test), average='micro')
      F1_score[i] = metrics.f1_score(y_test, dectree.predict(X_test), average='micro')
      
   
   # initial_pop[c].size = Xnew_n.shape[1]
   # initial_pop[c].acc=np.mean(s)
   # initial_pop[c].Fit=[(cl_num - initial_pop[c].size + 1 ) / cl_num, initial_pop[c].acc]
   return Xnew_n.shape[1], np.mean(s),[((cl_num - Xnew_n.shape[1] + 1 ) / cl_num), np.mean(s)], np.mean(Precision), np.mean(Recall), np.mean(F1_score) 
########### Dominates function to be used in Non-dominated sorting function#################
def Dominates(x,y):
    
    f1=False
    f2=False
    # flag=all(x >= y for i,j in (x,y))  and  any(x > y for i,j in (x,y))
    c1 = 0 
    c2 = 0
    for i in range(len(x)):
       if(x[i] >= y[i]):
           c1 += 1
       if(x[i] > y[i]):
           c2 += 1

    if(c1 == len(x)):
       f1=True
   
    if(c2 > 0):
       f2=True
   
    flag=f1 and f2   
    return flag

############# Non-dominated sorting function##################
def NonDominatedSorting(initial_pop):
    
    dFrame = pd.DataFrame(index=range(50),columns=range(1))
    nPop=len(initial_pop)
    for i in range(nPop):
        initial_pop[i].DominationSet=[]
        initial_pop[i].DominatedCount=0
        
    F=[]    
    for i in range (nPop):
        for j in range (i+1,nPop):
            p=initial_pop[i]
            q=initial_pop[j]
            if Dominates(p.Fit,q.Fit):
                # p.DominationSet=p.DominationSet.append(j)
                p.DominationSet.append(j)
                q.DominatedCount=q.DominatedCount+1
            if Dominates(q.Fit,p.Fit):
                # q.DominationSet=q.DominationSet.append(i)
                q.DominationSet.append(i)
                p.DominatedCount=p.DominatedCount+1                
            
            initial_pop[i]=p
            initial_pop[j]=q
        
        if initial_pop[i].DominatedCount==0:
          F.append(i)
          initial_pop[i].rank=0
    
    k=0
    dFrame.iloc[k,0]=F

    while(True):
        Q=[]
        for i in dFrame.iloc[k,0]:
            p=initial_pop[i]
            for j in p.DominationSet:
                q=initial_pop[j]
                q.DominatedCount=q.DominatedCount-1
                if q.DominatedCount==0:
                    Q.append(j)
                    q.rank=k+1
                initial_pop[j]=q
        if not Q:
            break;
            
        k=k+1
        dFrame.iloc[k,0]=Q
                    
    return initial_pop,F,dFrame


################  Calculate Distance######################

def CalCrowdingDistance(initial_pop, FrontsList):
    
 for s in range(int(FrontsList.shape[0])):  
   FitVector=[]
   for dd in range (int(len(FrontsList[0][s]))):
      FitVector.append(initial_pop[FrontsList[0][s][dd]].Fit) ###0=i
   nObj=len(FitVector[0])
   n=len(FitVector)
   d=np.zeros((n,nObj))
   for b in range (nObj):
      Fj=[]
      for l in range (n):
          Fj.append(FitVector[l][b])
      yy=np.sort(Fj)
      indexyy=(yy).argsort()[::-1]
      d[indexyy[0],b]=float('inf')
      for o in range (1,n-1):
         d[indexyy[o],b]=abs(yy[o+1]-yy[o-1])/abs(yy[0]-yy[-1])
      d[indexyy[-1],b]=float('inf')
   for i in range (n):
      initial_pop[FrontsList[0][s][i]].CrowdingDistance=np.sum(d[i,:])
         
         
 return initial_pop



################  Sort Population######################

def SortPopulation(initial_pop):
    
    # Sort based on crowding distance
    CD=[]
    RSO=[]
    for i in range (len(initial_pop)):
        CD.append(initial_pop[i].CrowdingDistance)
    # sorted(CD, reverse=True)
    CDSO=np.array(CD).argsort()[::-1]
    import operator
    initial_pop.sort(key=operator.attrgetter('CrowdingDistance'), reverse=True)
    # Sort based on ranks
    for i in range (len(initial_pop)):
        RSO.append(initial_pop[i].rank)
    RSO = np.array(RSO).argsort()
    initial_pop.sort(key=operator.attrgetter('rank'))
    
    
    FitVector=[]
    MaxAcc=[]
    Selected=[]
    for dd in range (len(initial_pop)):
        if (initial_pop[dd].rank==0):
           FitVector.append(initial_pop[dd].Fit)
           MaxAcc.append(initial_pop[dd].acc)
    MaximumAcc=max(MaxAcc)
    for dd in range (len(initial_pop)):
        if ((initial_pop[dd].rank==0) and (initial_pop[dd].acc==MaximumAcc)):
            Selected=initial_pop[dd].cl
            X1=initial_pop[dd].X
            y1=initial_pop[dd].Y
            P1=initial_pop[dd].P
            R1=initial_pop[dd].R
            F1=initial_pop[dd].FS
    return initial_pop, FitVector,Selected, MaximumAcc, X1, y1, P1, R1, F1

############# Main body of MDES################
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
opt=0
MaximumAccuracy=0 
beta=0.1
t0=time()
cl_num=10
from ypstruct import struct
nPop=cl_num
initial_solutions = struct(size=None, cl=None, acc=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, X=None, Y=None, P=None, R=None, FS=None)
initial_pop = initial_solutions.repeat(nPop)
th=np.full((1, cl_num), 0.5)      # setup initial threshold
th1=np.zeros(cl_num) 

# we want to cluster features to cl_num number of clusters 
Xnt=Xn.T
from sklearn.cluster import KMeans


kmeans=KMeans(n_clusters=cl_num).fit(Xnt)
cluster_labels=kmeans.labels_
cln=pd.DataFrame(cluster_labels)
cln.columns=['labels']
Weights=cl_num*[1/cl_num]
ClustersVector=np.array(range(cl_num))
 
sh=1
for c in range(nPop):
       mask=np.random.choice(ClustersVector,size=sh,replace=False, p=Weights)
       initial_pop[c].cl=mask
       subset=np.zeros(len(mask))

       for i in range(len(mask)):
          subset[i]=cln.labels[cln.labels.eq(mask[i])].sample().index.values

#construct the respective matrix
       matn=np.zeros((df.shape[0],len(mask)+1))
       for i in range(len(mask)):
          matn[:,i]=Xn.iloc[:,int(subset[i])]
       matn[:,len(mask)]=yn 
       Xnew_n=matn[:,0:len(mask)]
       ynew_n=matn[:,len(mask)]
       initial_pop[c].X=Xnew_n
       initial_pop[c].Y=ynew_n
       
       [initial_pop[c].size,initial_pop[c].acc,initial_pop[c].Fit, initial_pop[c].P, initial_pop[c].R, initial_pop[c].FS ]=Fitness(Xnew_n,ynew_n)
       sh=sh+1
####### Specifying ranks to solutions    
####### FrontsList contain the fronts' members including paretofront and the rest
[initial_pop, F, FrontsList]=NonDominatedSorting(initial_pop)
FrontsList=FrontsList[~FrontsList.isnull().any(axis=1)]
########Crowding Distance Calculation#################  
initial_pop=CalCrowdingDistance(initial_pop, FrontsList)
#########Sort population based on rank and crowding distance#################     
[initial_pop,ParetoFront, Selected,MaximumAcc, X1, Y1, P1, R1, F1]=SortPopulation(initial_pop)
MaximumAccuracy=MaximumAcc
members=len(Selected)
print ("Iteration",  "  ", 0,"      ","Number of F0 members", "  ",   len(ParetoFront)  )

MaxIt=200
for es in range (MaxIt):

   from ypstruct import struct
   nPop1=cl_num
   initial_solutions = struct(size=None, cl=None, acc=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, X=None, Y=None, P=None, R=None, FS=None)
   initial_pop2 = initial_solutions.repeat(nPop1)
   th1=np.zeros(cl_num) 

   sh=1
   for c in range(nPop1):
      
       mask=np.random.choice(ClustersVector,size=sh,replace=False, p=Weights)
       initial_pop2[c].cl=mask
      # subset_size=sum(th1<th)
# construct one solution
       subset=np.zeros(len(mask))

       for i in range(len(mask)):
         subset[i]=cln.labels[cln.labels.eq(mask[i])].sample().index.values

#construct the respective matrix
       matn=np.zeros((df.shape[0],len(mask)+1))
# df_ = pd.DataFrame(index=61, columns=cl_num+1)
       for i in range(len(mask)):
           matn[:,i]=Xn.iloc[:,int(subset[i])]
       matn[:,len(mask)]=yn 
       Xnew_n=matn[:,0:len(mask)]
       ynew_n=matn[:,len(mask)]
       initial_pop2[c].X=Xnew_n
       initial_pop2[c].Y=ynew_n
           
       [initial_pop2[c].size,initial_pop2[c].acc,initial_pop2[c].Fit, initial_pop2[c].P, initial_pop2[c].R, initial_pop2[c].FS]=Fitness(Xnew_n,ynew_n)
       sh=sh+1 

   initial_pop=initial_pop+initial_pop2   
####### Specifying ranks to solutions    
####### FrontsList contain the fronts' members including paretofront and the rest
   [initial_pop, F, FrontsList]=NonDominatedSorting(initial_pop)
   FrontsList=FrontsList[~FrontsList.isnull().any(axis=1)]
########Crowding Distance Calculation#################  
   initial_pop=CalCrowdingDistance(initial_pop, FrontsList)
#########Sort population based on rank and crowding distance#################     
   [initial_pop, ParetoFront, Selected, MaximumAcc, X1, Y1, P1, R1, F1]=SortPopulation(initial_pop)
   
   
   ####Truncate
   initial_pop=initial_pop[0:nPop]
  
   if MaximumAcc>MaximumAccuracy:
      print(pd.DataFrame(X1).head()) 
      opt=opt+1
      Xf=X1
      yf=Y1
      PRE=P1
      RE=R1
      FSC=F1
      print(pd.DataFrame(Xf).head())
      MaximumAccuracy=MaximumAcc
      members=len(Selected)
      for i in Selected:
              
                  Weights[i]=Weights[i]+beta*(1-Weights[i])
              
      Weights=np.array(Weights)/sum(np.array(Weights))  


   print ("Iteration",  "  ", es,"      ","Number of F0 members", "  ",  len(ParetoFront)   )



##################################################





######## Separate code for calculation metrics of any given dataset###############
from sklearn import metrics 
Precision=np.zeros(10)
Recall=np.zeros(10)
F1_score=np.zeros(10)
s=np.zeros(10)
nm=np.zeros(10)
 

for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
  dectree = tree.DecisionTreeClassifier()
  dectree.fit(X_train,y_train)
  s[i]=dectree.score(X_test,y_test)
   # nm[i]=normalized_mutual_info_score(dectree.predict(X_test),y_test)
  cm=confusion_matrix(y_test,dectree.predict(X_test))
  Precision[i] = metrics.precision_score(y_test, dectree.predict(X_test),average='micro')
  Recall[i] = metrics.recall_score(y_test, dectree.predict(X_test),average='micro')
  F1_score[i] = metrics.f1_score(y_test, dectree.predict(X_test),average='micro')
  # svm_linear=svm.SVC(C=50,kernel="linear")
  # svm_linear.fit(X_train,y_train)
  # s[i]=svm_linear.score(X_test,y_test)

  # cm=confusion_matrix(y_test,svm_linear.predict(X_test))
#   p_final[i]= cm[1,1] / (cm[0,1]+cm[1,1])
#   r_final[i]= cm[1,1] / (cm[1,1]+cm[1,0])
# pre=np.mean(p_final) 
# rec=np.mean(r_final) 
# fscore=2 * ((pre*rec) / (pre+rec))
np.mean(s)
np.mean(Precision)
np.mean(Recall)
np.mean(F1_score)
 






