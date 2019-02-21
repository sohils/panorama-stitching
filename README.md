Panorama Stitching
======

## How to run the panorama stitching
Execute the file `panorama.py` using the following command:
```
python panorama.py --im1=<path to image 1> --im2=<path ot image 2>
```

To run it on the default images present in the `data` folder, just run the following command:
```
python panorama.py
```

(Beware: It will take a couple of minutes to execute on very large resolution images)

The final stitched image is saved under `results/panoImg.png'.

## How does it work
Have you ever wondered how Google or Apple are able to stitch two or more photos together to create a panorama picture or even a 360 degree field of view picture?

It's easy! ...(Sort of)

Steps to create a panorama stitching:

1. _Find the feature points_ in each image that uniquely describe key points that may be identifiable in multiple views of the same object/feature. In this project, I implented a lean version of the **BRIEF** (Binary Robust Independent Elementary Features) algorithm that essentially creates a fingerprint of key points (certain edges and corners).
2. _Match these feature points between the two images_. Using these fingerprints of important points, we cross reference them between similar fingerprints from another image to identify the closest match.
3. Use of **RANSAC** (Random sample consensus) to identify the best homogrphy transformation between the two views.
4. Apply the homography to the second image and combine the two images to create a seamless panorama. 

To study the methods talked about BRIEF-ly ;) above, I recommend you go through the resources below.
## References
[1] BRIEF: Paper https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf

[2] RANSAC: https://en.wikipedia.org/wiki/Random_sample_consensus

[3] If music videos are your thing, checkout RANSAC: https://www.youtube.com/watch?v=1YNjMxxXO-E

[4] Derivation of the Homography estimation used: https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf