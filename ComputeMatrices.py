# you should fill in the functions in this file,
# do NOT change the name, input and output of these functions

import numpy as np
import time
import matplotlib.pyplot as plt

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

    # replacing the zeros minimum number acceptable, 
    # anything smaller will produce a nan in the operations
    # std_vec[std_vec == 0] = 2.22*(10**(-162)) 
    #creating std matrix for element wise division with cov_mat
    std_mat =  np.outer(std_vec,std_vec)
  
    M = cov_mat / std_mat #matrix without nan's

    return M

def main():
    print ('starting comparing distance computation .....')
    np.random.seed(100)
    params = range(10,141,10)   # different param setting
    nparams = len(params)       # number of different parameters

    perf_dist_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_dist_cool = np.zeros([10,nparams])
    perf_corr_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_corr_cool = np.zeros([10,nparams])

    counter = 0

    for ncols in params:
        nrows = ncols * 10

        print ("matrix dimensions: ", nrows, ncols)

        for i in range(10):
            X = np.random.rand(nrows, ncols)   # random matrix

            # compute distance matrices
            st = time.time()
            dist_loop = compute_distance_naive(X)
            et = time.time()
            perf_dist_loop[i,counter] = et - st              # time difference

            st = time.time()
            dist_cool = compute_distance_smart(X)
            et = time.time()
            perf_dist_cool[i,counter] = et - st

            assert np.allclose(dist_loop, dist_cool, atol=1e-06) # check if the two computed matrices are identical all the time

            # compute correlation matrices
            st = time.time()
            corr_loop = compute_correlation_naive(X)
            et = time.time()
            perf_corr_loop[i,counter] = et - st              # time difference

            st = time.time()
            corr_cool = compute_correlation_smart(X)
            et = time.time()
            perf_corr_cool[i,counter] = et - st

            assert np.allclose(corr_loop, corr_cool, atol=1e-06) # check if the two computed matrices are identical all the time

        counter = counter + 1

    mean_dist_loop = np.mean(perf_dist_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_dist_cool = np.mean(perf_dist_cool, axis = 0)
    std_dist_loop = np.std(perf_dist_loop, axis = 0)      # standard deviation
    std_dist_cool = np.std(perf_dist_cool, axis = 0)

    plt.figure(1)
    plt.errorbar(params, mean_dist_loop[0:nparams], yerr=std_dist_loop[0:nparams], color='red',label = 'Loop Solution for Distance Comp')
    plt.errorbar(params, mean_dist_cool[0:nparams], yerr=std_dist_cool[0:nparams], color='blue', label = 'Matrix Solution for Distance Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Distance Computation Methods')
    plt.legend()
    plt.savefig('CompareDistanceCompFig.pdf')
  #  plt.show()    # uncomment this if you want to see it right way
    print ("result is written to CompareDistanceCompFig.pdf")

    mean_corr_loop = np.mean(perf_corr_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_corr_cool = np.mean(perf_corr_cool, axis = 0)
    std_corr_loop = np.std(perf_corr_loop, axis = 0)      # standard deviation
    std_corr_cool = np.std(perf_corr_cool, axis = 0)

    plt.figure(2)
    plt.errorbar(params, mean_corr_loop[0:nparams], yerr=std_corr_loop[0:nparams], color='red',label = 'Loop Solution for Correlation Comp')
    plt.errorbar(params, mean_corr_cool[0:nparams], yerr=std_corr_cool[0:nparams], color='blue', label = 'Matrix Solution for Correlation Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Correlation Computation Methods')
    plt.legend()
    plt.savefig('CompareCorrelationCompFig.pdf')
   # plt.show()    # uncomment this if you want to see it right way
    print ("result is written to CompareCorrelationCompFig.pd")
if __name__ == "__main__": main()
