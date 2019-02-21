import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import random
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF

    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor

    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                            patch and are each (nbits,) vectors. 
    '''
    #############################
    # generates a Random Gaussian Pattern which is centered
    getRandomPattern = get_truncated_normal()
    compareX = (getRandomPattern.rvs(81)*patch_width) + getRandomPattern.rvs(81)
    compareY = (getRandomPattern.rvs(81)*patch_width) + getRandomPattern.rvs(81)

    # # This generates a truly random pattern (Performs Better)
    # points = np.zeros((nbits*2, 2))
    # for i in range(nbits*2):
    #     points[i,:]=random.sample(range(0,81),2)
    # points = np.unique(points,axis=0)
    # points = points[:nbits,:]
    # compareX = points[:,0]
    # compareY = points[:,1]
    # np.save("results/testPattern", [compareX, compareY])
    return compareX, compareY

def get_truncated_normal(mean=4, sd=(9/5), low=0, upp=8):
    # generate a truncated continous random variable 
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

def computeBrief(im, gaussian_pyramid, locsDoG,
    compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the 
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.
    
    
     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################
    patch_width = 9

    locs = []
    desc = []
    for point in range(locsDoG.shape[0]):
       
        if(  # Eliminating points close to the edges
            (locsDoG[point, 0] - int(patch_width/2))<0 or
            (locsDoG[point, 1] - int(patch_width/2))<0 or 
            (locsDoG[point, 0] + int(patch_width/2))>=im.shape[1] or
            (locsDoG[point, 1] + int(patch_width/2))>=im.shape[0] 
            ):
            continue
        locs.append(locsDoG[point,:])
        desc_point = np.zeros(compareX.size)

        for comparison in range(compareX.size):
            # Converting linear indices to patch X (row, column) and patch Y (row, column) 
            # wrt to point X and point Y
            move_x_r = int(compareX[comparison] / 9) - int(patch_width/2)
            move_x_c = int(compareX[comparison] % 9) - int(patch_width/2)
            move_y_r = int(compareY[comparison] / 9) - int(patch_width/2)
            move_y_c = int(compareY[comparison] % 9) - int(patch_width/2)

            if(gaussian_pyramid[locsDoG[point,1] + move_x_r, locsDoG[point,0] + move_x_c, locsDoG[point,2]] 
                > gaussian_pyramid[locsDoG[point,1] + move_y_r, locsDoG[point,0] + move_y_c, locsDoG[point,2]]):
                # Comparing point X and point Y to create a unique fingerprint of the neighbourhood of the point 
                desc_point[comparison] = 1

        desc.append(desc_point)
    
    # list to ndarry conversion
    desc = np.stack(desc, axis = 0)
    locs = np.stack(locs, axis = 0)

    return locs, desc

def briefLite(im):
    '''
    INPUTS
    im - gray image with values between 0 and 1

    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    ###################
    # Computer DoG
    locsDoG, gaussian_pyramid = DoGdetector(im)

    # Perform BRIEF to get the descriptors of feature points
    locs, desc = computeBrief(im, gaussian_pyramid, locsDoG, compareX, compareY)
    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    # find the distance between all pairs of points
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')

    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)

    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)

    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    # draw two images side by side
    fig = plt.figure()
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r',linewidth=0.5)
        plt.plot(x,y,'g.',linewidth=0.5)
    plt.show()
    
    

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    # # test briefLite
    # # im = cv2.imread('../data/model_chickenbroth.jpg')
    # # locs, desc = briefLite(im)  
    # # fig = plt.figure()q2_
    # # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    # # plt.plot(locs[:,0], locs[:,1], 'r.')
    # # plt.draw()
    # # plt.waitforbuttonpress(0)
    # # plt.close(fig)
    # # test matches
    im1 = cv2.imread('../data_proj/dinoR0023.png')
    im2 = cv2.imread('../data_proj/dinoR0024.png')
    print(im1.shape,im2.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)