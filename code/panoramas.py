import argparse

import math
import cv2
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt

from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    Warps im2 into im1 reference frame using the provided warpH() function

    INPUT
        im1   - first image
        im2   - second image that is warped onto the im1
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends im1 and warped im2 and outputs the panorama image
    '''
    #######################################
    # Shape of warped image
    warped_im2_shape=(im1.shape[1]+800,im1.shape[0]+100)
    warped_im2_depth = im1.shape[2]

    # Warp im2
    warp_m = cv2.warpPerspective(im2, H2to1, warped_im2_shape)
    im1 = np.hstack((im1,np.zeros((im1.shape[0],800,im1.shape[2]))))
    im1 = np.vstack((im1,np.zeros((100,im1.shape[1],im1.shape[2]))))
    mask_im1 = distance_transform_edt(im1)
    mask_im2 = distance_transform_edt(warp_m)

    pano_im = (mask_im1*im1 + mask_im2*warp_m)/(mask_im1+mask_im2)

    return warp_m.astype(np.uint8)


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    im1_corners = getCorners(im1)
    im2_corners = getCorners(im2)

    w_im2_corners = np.vstack((im2_corners,np.ones((1,im2_corners.shape[1]))))
    warp_im2_corners = np.matmul(H2to1, w_im2_corners)
    warp_im2_corners = warp_im2_corners/warp_im2_corners[2,:]

    # Determine the translation in x and y depending on the position of im1 wrt im2
    tx = 0 if min(im1_corners[0,:])<min(warp_im2_corners[0,:]) else abs(min(warp_im2_corners[0,:]))
    ty = 0 if min(im1_corners[1,:])<min(warp_im2_corners[1,:]) else abs(min(warp_im2_corners[1,:]))

    all_corners = np.hstack((im1_corners,warp_im2_corners[:2,:]))
    
    width = im1.shape[1]
    
    # Find max height and width of stitched image
    pano_width = math.ceil(max(all_corners[0,:])-min(all_corners[0,:]))
    pano_height = math.ceil(max(all_corners[1,:])-min(all_corners[1,:]))

    # Scale final image according to desired width of the panaromic image
    scale = width/pano_width
    pano_size = (np.ceil(pano_width*scale).astype(int),np.ceil(pano_height*scale).astype(int))
    
    # Scale and translation factor for both images so as to fit them in the panaroma
    scale_M = np.array([[scale,0,0],[0,scale,0],[0,0,1]])
    M = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    M = np.matmul(scale_M,M)

    # Warp im1
    warp_im1 = cv2.warpPerspective(im1, M, pano_size)

    # Warp im2
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), pano_size)

    # Mask for im1 and im2
    mask_im1 = distance_transform_edt(warp_im1)
    mask_im2 = distance_transform_edt(warp_im2)

    # Merging the masks to obtain a smooth panoramic image
    pano_im = (mask_im1*warp_im1 + mask_im2*warp_im2)/(mask_im1+mask_im2)

    return pano_im.astype(np.uint8)

def getCorners(im):
    '''
    Returns an ndarray of the x and y coordinates of the 4 corners of the image
    '''
    im_corners = np.hstack(
        (
            np.vstack((0,0)),
            np.vstack((0,im.shape[0]-1)),
            np.vstack((im.shape[1]-1,im.shape[0]-1)),
            np.vstack((im.shape[1]-1,0))
        )
    )
    return(im_corners)

def generatePanorama(im1, im2):
    # computes keypoints and descriptors for both the images
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)

    # finds putative feature correspondences by matching keypoint descriptors
    matches = briefMatch(desc1, desc2)

    # estimates a homography using RANSAC
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

    # warps one of the images with the homography so that they are aligned and then overlays them.
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    # save and display panaroma
    cv2.imwrite('../results/panoImg.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args):
    im1 = cv2.imread(args.im1)
    im2 = cv2.imread(args.im2)

    # Compute BREIF descriptors
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)

    # Match descriptors
    matches = briefMatch(desc1, desc2)
    
    # Estimate best Homogrpahy H
    H2to1,_ = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    
    # Stitch images together
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    # Save and show
    cv2.imwrite('../results/panoImg.jpg', pano_im)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--im1", dest="im1",
                    help="Path of image 1", default="../data/incline_L.png")
    parser.add_argument("-j", "--im2", dest="im2",
                    help="Path of image 2", default="../data/incline_R.png")
    args = parser.parse_args()
    main(args)