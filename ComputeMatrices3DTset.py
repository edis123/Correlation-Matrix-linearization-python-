# you should fill in the functions in this file,
# do NOT change the name, input and output of these functions

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn import datasets
from prettytable import PrettyTable

titles = ["DataSet","Dist_Loop_Time(s)","Dist_Smart_Time(s)","Correlation_Mat_Loop_Time(s)", "Correlation_Mat_Smart_Time(s)"]
data_table = PrettyTable(titles)
iris = datasets.load_iris()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()


data_sets =[iris,breast_cancer,digits] 

# first function to fill, compute distance matrix using loops
def compute_distance_naive(X):
    N = X.shape[0]      # num of rows
    D = X[0].shape[0]   # num of cols
    
    M = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            x_i = X[i,:]
            x_j = X[j,:]
            distance = (np.sum((x_i -x_j) **2)) ** 0.5
            # or distance = np.linalg.norm(x_i-x_j)
            
            M[i,j] = distance

    return M

# second function to fill, compute distance matrix without loops
def compute_distance_smart(X):
    X1= np.array(X)
    # N = X.shape[0]
    # D = X[0].shape[0] 
     
    # squared Euclidean norm
    squared_norms = np.sum(X1**2, axis = 1, keepdims=True)
    squared_norms = np.array(squared_norms)
    # applyinng formula / avoiding negative roots
    M = np.sqrt(np.maximum(0,(squared_norms - (2*(X1 @ X1.T)) + squared_norms.T)))
    
    return M

# third function to fill, compute correlation matrix using loops
def compute_correlation_naive(X):
    X = np.array(X)
    N = X.shape[0]
    D = X[0].shape[0]
    M = np.zeros((D, D))

    means_X = np.mean(X, axis=0)
   
    for i in range(D):
        for j in range(D):
            x_i = X[:,i]
            x_j = X[:,j]

           # applying cov_ij/(var_xi^0.5 * var_xj^0.5)
            var_xi = np.sum((x_i - means_X[i])**2)/ (N-1)
            var_xj = np.sum((x_j - means_X[j])**2)/ (N-1)
            cov_ij = np.sum((x_i- means_X[i]) * (x_j - means_X[j])) / (N - 1)

            std1 = np.sqrt(var_xi)
            std2 = np.sqrt(var_xj)
            if(std1 == 0):
                std1 = 2.22*(10**(-162)) # assigninng smallest nr possible to avoid nan
            if(std2 == 0):
                std2 = 2.22*(10**(-162)) 
            r_ij = cov_ij / (std1 * std2)

            M[i, j] = r_ij


    return M

# fourth function to fill, compute correlation matrix without loops
def compute_correlation_smart(X):
    X = np.array(X)
    X = X.transpose();
    N = X.shape[0]
    D = X[0].shape[0]
    

    means = np.mean(X, axis=1, keepdims=True);
    centered_data = X - means; #need this vor cov
   
    cov_mat = np.dot(centered_data, centered_data.T) / (N-1)
    #getting std vector from one of the diagonale
    std_vec = np.sqrt(np.diag(cov_mat)) # 1 diagonale is cov and the other is var
    # minimum number acceptable, anything smaller will produce a nan in the operations
    #std_vec[std_vec == 0] = 2.22*(10**(-162)) 
    #creating std matrix for element wise division with cov_mat
    std_mat =  np.outer(std_vec,std_vec)
  
    M = cov_mat / std_mat #matrix without nan's

    return M

def main():
    print ('Starting computation .....')
    datasets=np.array([])
    time_in_sec =np.array([])
    computation=np.array([])
    for i in range(len(data_sets)):
          filename ="";
          if hasattr(data_sets[i], "filename"):
              filename = data_sets[i].filename
          else:
              filename = "digits.csv"   #digits has no filename attribute

          print ("DataSet: ",filename," Matrix Dimensions: ",  data_sets[i].data.shape[0], data_sets[i].data[0].shape[0])
    
            # compute distance matrices
          st = time.time()
          compute_distance_naive(data_sets[i].data)
          et = time.time()
          t1 = et - st              # time difference
        
          st = time.time()
          compute_distance_smart(data_sets[i].data)
          et = time.time()
          t2 = et - st

            # compute correlation matrices
          st = time.time()
          print(compute_correlation_naive(data_sets[i].data))
          et = time.time()
          t3 = et - st              # time difference
          
          st = time.time()
          compute_correlation_smart(data_sets[i].data)
          et = time.time()
          t4 = et - st
          
          data_table.add_row([filename,t1,t2,t3,t4])
          
    print(data_table)
    print(iris.DESCR)

if __name__ == "__main__": main()
