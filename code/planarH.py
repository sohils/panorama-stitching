import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import random
import matplotlib.pyplot as plt
from BRIEF import briefLite,briefMatch,plotMatches

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    #############################
    H2to1 = None

    A = np.zeros((2* p1.shape[1], 9))
    for i in range(p1.shape[1]):
        # Creating a system of 8 equations
        A[i*2 + 1, :] = [p2[0,i], p2[1,i], 1, 0, 0, 0, -p1[0,i]*p2[0,i], -p1[0,i]*p2[1,i], -p1[0,i]]
        A[i*2 + 0, :] = [0, 0, 0, -p2[0,i], -p2[1,i], -1, p1[1,i]*p2[0,i], p1[1,i]*p2[1,i], p1[1,i]]
    
    # Last row of V will give us result of minimization wrt above equations
    U,S,V = np.linalg.svd(np.matmul(A.T,A), False)
    H = np.reshape(V[8,:],(3,3))

    # Divide by scaling factor
    H = H/H[2,2]
    
    H2to1 = H

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    n_inliers = 0
    number_of_matches = matches.shape[0]

    x_coordinates = np.zeros((2,number_of_matches))
    u_coordinates = np.zeros((2,number_of_matches))

    for i in range(number_of_matches):
        x_coordinates[:,i] = locs1[matches[i,0],0:2]
        u_coordinates[:,i] = locs2[matches[i,1],0:2]

    u_coordinates = np.vstack((u_coordinates,np.ones(u_coordinates.shape[1])))

    for it in range(num_iter):
        # Perform for fixed number of iterations
        sample_x_coordinates = np.zeros((2,4))
        sample_u_coordinates = np.zeros((2,4))

        rand_list = random.sample(range(number_of_matches),4)
        for i in range(4):
            # Sample 4 random pairs of corresponding points 
            sample_x_coordinates[:,i] = locs1[matches[rand_list[i],0],0:2]
            sample_u_coordinates[:,i] = locs2[matches[rand_list[i],1],0:2]

        # Compute the Homography based on these points
        maybeBestH = computeH(sample_x_coordinates, sample_u_coordinates)
        
        # Verify how good the H from the 4 random pairs
        computed_x = np.matmul(maybeBestH, u_coordinates)
        computed_x = computed_x/computed_x[2,:]
        computed_x = computed_x[:2,:]
        ssd = np.sqrt(np.sum((x_coordinates - computed_x)**2,axis=0))

        # Find the number of inliers
        inliers_pts = np.argwhere(ssd<=2)
        n_inliers_it = inliers_pts.size
        # print(n_inliers_it)
        if(n_inliers_it >  n_inliers):
            # More the inliers, better the estimation
            n_inliers = n_inliers_it
            bestH = maybeBestH
            bestinliers = inliers_pts
    
    return bestH,matches[bestinliers.flatten()]