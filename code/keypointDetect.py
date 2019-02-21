import numpy as np
import cv2
import matplotlib.pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    '''
    Produces Gaussian Pyramid of an image

    [inputs]
    * im        - Image ndarray of size [imH, imW, 3] in cv2 default format (BGR)
    * sigma     - 
    * k         -
    * levels    - Levels of the pyramid specifying the blur
    
    [outputs]
    * Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    '''
    ################
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid

    [inputs]
    * Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    * levels           - The levels of the pyramid specifying the blur
    
    [outputs]
    * DoG Pyramid  - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    * DoG levels   -  the levels of the DoG
    '''
    ################
    DoG_pyramid = []
    for level in range(len(levels) - 1):
        # subtracting subsequent gaussians of the image to create a DoG
        DoG_pyramid.append(gaussian_pyramid[:,:,level+1] - gaussian_pyramid[:,:,level])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature, a matrix of the same size where each point contains the
    curvature ratio R for the corresponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    ################
    principal_curvature = None
    DoG_pyramid_sobels_dxx = []
    DoG_pyramid_sobels_dyy= []
    DoG_pyramid_sobels_dxy = []
    for level in range(DoG_pyramid.shape[2]):
        # cv2 applies Sobel only only on one 2D array at a time.
        DoG_pyramid_sobels_dxx.append(cv2.Sobel(DoG_pyramid[:,:,level], ddepth= -1, dx=2, dy=0, borderType=cv2.BORDER_CONSTANT))
        DoG_pyramid_sobels_dyy.append(cv2.Sobel(DoG_pyramid[:,:,level], ddepth= -1, dx=0, dy=2, borderType=cv2.BORDER_CONSTANT))
        DoG_pyramid_sobels_dxy.append(cv2.Sobel(DoG_pyramid[:,:,level], ddepth= -1, dx=1, dy=1, borderType=cv2.BORDER_CONSTANT))
    
    # convert the list of sobels to an ndarray of size (imH, imW, len(levels) - 1)
    DoG_pyramid_sobels_dxx = np.stack(DoG_pyramid_sobels_dxx, axis=-1)
    DoG_pyramid_sobels_dyy = np.stack(DoG_pyramid_sobels_dyy, axis=-1)
    DoG_pyramid_sobels_dxy = np.stack(DoG_pyramid_sobels_dxy, axis=-1)

    # flatten to quicken calculations
    DoG_pyramid_sobels_dxx = DoG_pyramid_sobels_dxx.flatten()
    DoG_pyramid_sobels_dyy = DoG_pyramid_sobels_dyy.flatten()
    DoG_pyramid_sobels_dxy = DoG_pyramid_sobels_dxy.flatten()

    # Redunduncy check for 
    assert(DoG_pyramid_sobels_dxx.shape == DoG_pyramid_sobels_dyy.shape)
    assert(DoG_pyramid_sobels_dxx.shape == DoG_pyramid_sobels_dxy.shape)

    principal_curvature = np.zeros(DoG_pyramid_sobels_dxx.size)

    for i in range(DoG_pyramid_sobels_dxx.size):
        # create a hessian matrix
        hessian = np.array(
            [
                DoG_pyramid_sobels_dxx[i],
                DoG_pyramid_sobels_dxy[i],
                DoG_pyramid_sobels_dxy[i],
                DoG_pyramid_sobels_dyy[i]
                ]
                ).reshape(2,2)
        det = np.linalg.det(hessian)

        # pc = trace(H)^2/|H| (at every point)
        principal_curvature[i] = (np.trace(hessian)**2 / det) if det!=0 else 0

    principal_curvature = principal_curvature.reshape(DoG_pyramid.shape[0], DoG_pyramid.shape[1], DoG_pyramid.shape[2])
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    ##############
    locsDoG = None
    kernel = np.ones((3,3),np.uint8)

    maxima_level = []
    minima_level = []
    for level in range(len(DoG_levels)):
        # Identify for neighbourhood (of 8) maxima and maxima points in the same level 
        maxima_in_level = cv2.compare(DoG_pyramid[:,:,level], cv2.dilate(DoG_pyramid[:,:,level],kernel,iterations = 1), cmpop= cv2.CMP_EQ)/255.0
        minima_in_level = cv2.compare(DoG_pyramid[:,:,level], cv2.erode(DoG_pyramid[:,:,level],kernel,iterations = 1), cmpop= cv2.CMP_EQ)/255.0
        maxima_level.append(maxima_in_level)
        minima_level.append(minima_in_level)

    maxima_level = np.stack(maxima_level, axis = -1)
    minima_level = np.stack(minima_level, axis = -1)

    # Experimental code
    maxima_across_level = []
    minima_across_level = []
    for col in range(DoG_pyramid.shape[0]):
        # Identify for neighbourhood (of 2) maxima and maxima points across levels 
        maxima_in_height = cv2.compare(DoG_pyramid[col,:,:].T,cv2.dilate(DoG_pyramid[col,:,:].T,np.array([1,1,1])), cv2.CMP_EQ).T/255.0
        minima_in_height = cv2.compare(DoG_pyramid[col,:,:].T,cv2.erode(DoG_pyramid[col,:,:].T,np.array([1,1,1])), cv2.CMP_EQ).T/255.0
        maxima_across_level.append(maxima_in_height)
        minima_across_level.append(minima_in_height)

    maxima_across_level = np.stack(maxima_across_level, axis=-1)
    minima_across_level = np.stack(minima_across_level, axis=-1)
    maxima_across_level = np.moveaxis(maxima_across_level, -1, 0)
    minima_across_level = np.moveaxis(minima_across_level, -1, 0)
    
    assert(maxima_level.shape == maxima_across_level.shape)
    assert(minima_level.shape == minima_across_level.shape)

    # Combine the maximas and minimas in and across level to get extrema in neighbourhood of 10 (8 in plane and 2 above and below)
    maximas = maxima_level*maxima_across_level
    minimas = minima_level*minima_across_level

    # All extrema
    extrema = minimas + maximas

    # Extract the edge points
    DoG_extrema = extrema * DoG_pyramid
    principal_curvature_extrema = extrema * principal_curvature

    # Thresholding to find the points of intrest
    DoG_extrema_coordinates = np.argwhere((DoG_extrema!=0) & (abs(DoG_extrema)>th_contrast))
    principal_curvature_coordinates = np.argwhere(abs(principal_curvature_extrema)<th_r) 

    locsDoG = np.array([x for x in set(tuple(x) for x in DoG_extrema_coordinates) 
                & set(tuple(x) for x in principal_curvature_coordinates)])
    locsDoG[:,0], locsDoG[:,1] = locsDoG[:,1], locsDoG[:,0].copy()
    
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS
        im           - Grayscale image with range [0,1].
        sigma0       - Scale of the 0th image pyramid.
        k            - Pyramid Factor.  Suggest sqrt(2).
        levels       - Levels of pyramid to construct. Suggest -1:4.
        th_contrast  - DoG contrast threshold.  Suggest 0.03.
        th_r         - Principal Ratio threshold.  Suggest 12.

    OUTPUTS
        locsDoG       -  N x 3 matrix where the DoG pyramid achieves a local extrema
                        in both scale and space, and satisfies the two thresholds.

        gauss_pyramid -  A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    locsDoG = None
    gaussian_pyramid = None

    # Create the Gaussian Pyramid of the image
    gaussian_pyramid = createGaussianPyramid(im)
    
    # Compute the Difference of Gaussian
    DoG_pyramid, DoG_levels = createDoGPyramid(gaussian_pyramid, levels)

    # Compute the principal curvature matrix
    principal_curvature = computePrincipalCurvature(DoG_pyramid)

    # Lower bound for DoG
    th_contrast = 0.03

    # Upper bound for principal curvature
    th_r = 12

    # Perform edge supression to obtain feature points
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast, th_r)

    return locsDoG, gaussian_pyramid


def showPoints(locsDoG, img_name):
    '''
    Displays the image with its feature points

    INPUTS
        locsDoG  - N x 3 matrix where the DoG pyramid achieves a local extrema
                   in both scale and space, and satisfies the two thresholds.
        img_name - string specifying the location of the image
    '''
    ################
    img = plt.imread(img_name)
    plt.imshow(img)
    plt.scatter(locsDoG[:,0],locsDoG[:,1],c='g',s=5)
    plt.show()