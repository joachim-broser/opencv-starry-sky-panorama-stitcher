#!/bin/python3.10

import os.path

import cv2 as cv
import numpy as np
from skimage import exposure


def optimize_img_for_feature_detection(cv_image,prompt_dialog=False):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve ORB feature detection.

    Parameters
    ----------
    cv_image : np.ndarray(shape=(:,:,3), dtype=uint8)
        3-channel image.

    Returns
    -------
    img_adjusted : np.ndarray(shape=(:,:), dtype=uint8)
    	Grayscale image with improved contrast.
    """
    img_adjusted = cv_image.copy()
    img_adjusted = cv.cvtColor(img_adjusted, cv.COLOR_BGR2GRAY)

    img_adjusted = exposure.equalize_adapthist(img_adjusted, clip_limit=0.0125)
    img_adjusted = (img_adjusted * 255).astype(np.uint8)

    return img_adjusted

def adjust_black_and_white_point(img, black_and_white_point_tpl):
    """GIMP-like Histogram clipping
    """

    if black_and_white_point_tpl:
        black_point, white_point = black_and_white_point_tpl
        return ((np.clip(img, black_point, white_point) - black_point) * (255 / (white_point - black_point))).astype(
            np.uint8)
    else:
        return img

def compute_star_features(img, img_name, output_dir, n_brightest_stars = 500):
    """
    Detects stars by Canny Edge Detection, converts them into contours and these contours into keypoints.

    Parameters
    ----------
    img : np.ndarray(shape=(:,:,3), dtype=uint8)
        3-channel image.


    Returns
    -------
    img_features_dummy: cv2.detail.ImageFeatures
        Detected stars as an cv2.detail.ImageFeatures object, comprising descriptors and keypoints.
    """

    if False:
        # Find best Canny thresholds:
        def callback(x):
            print(x)

        # img = cv.imread('your_image.png', 0)  # read image as grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny = cv.Canny(img, 85, 255)

        cv.namedWindow('image')  # make a window with name 'image'
        cv.createTrackbar('L', 'image', 0, 255, callback)  # lower threshold trackbar for window 'image
        cv.createTrackbar('U', 'image', 0, 255, callback)  # upper threshold trackbar for window 'image

        while (1):
            numpy_horizontal_concat = np.concatenate((img, canny), axis=1)  # to display image side by side
            cv.imshow('image', numpy_horizontal_concat)
            k = cv.waitKey(1) & 0xFF
            if k == 27:  # escape key
                break
            l = cv.getTrackbarPos('L', 'image')
            u = cv.getTrackbarPos('U', 'image')

            canny = cv.Canny(img, l, u)

        cv.destroyAllWindows()

    if "Detect stars with canny edge":
        orig_image = img.copy()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(
            img_gray,
            201,  # first threshold for the hysteresis procedure.
            81  # second threshold for the hysteresis procedure.
        )

        cv.imwrite(os.path.join(output_dir, f"{img_name}___05_canny_edges.jpg"), edges)

        # cv.imshow("thresh", edges)

        contours, hierarchy = cv.findContours(
            edges,
            cv.RETR_TREE,  # mode
            cv.CHAIN_APPROX_NONE  # method
        )

        star_contours = cv.drawContours(
            orig_image.copy(),
            contours,
            -1,  # contourIdx: Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
            (0, 255, 0),
            2  # thickness
        )

        cv.imwrite(os.path.join(output_dir, f"{img_name}___06_contours.jpg"), star_contours)

        keypoints = list()

        # Get contour centers (= star centers)
        print(f"Detected {contours.__len__()} contours.")
        for c in contours:
            # compute the center of the contour
            M = cv.moments(c)

            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                # traceback.print_exc()
                cX, cY = int(np.mean(c[:, 0, 0])), int(np.mean(c[:, 0, 1]))
                if c.shape[1] != 1:
                    raise ValueError()

            # There are duplicate contours for unknown reason. Eliminate them
            if (cX, cY) in [(kp.pt[0], kp.pt[1]) for kp in keypoints]:
                continue

            keypoint_dummy = cv.KeyPoint()
            keypoint_dummy.angle = -1.0
            keypoint_dummy.class_id = -1
            keypoint_dummy.octave = 0
            keypoint_dummy.pt = (cX, cY)
            keypoint_dummy.response = 0.0
            keypoint_dummy.size = cv.contourArea(c)

            keypoints.append(keypoint_dummy)

            if False:
                # Debug
                if abs(cX - 100) < 2 and abs(cY - 77) < 2:
                    print(cX, cY)
                    # draw the contour and center of the shape on the image
                    copy = orig_image.copy()
                    cv.drawContours(copy, [c], -1, (0, 255, 0), 2)
                    # cv.circle(img_gray, (cX, cY), 7, (255, 255, 255), -1)
                    cv.putText(copy, "center", (cX - 20, cY - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # show the image
                    cv.imshow("Image", copy)
                    cv.waitKey(0)

        print(f"Created {keypoints.__len__()} keypoints out of {contours.__len__()} contours.")

    # To generate cv.detail.ImageFeatures, keypoints are not sufficient.
    # Descriptors won't be used later, but they are needed as well for integrity of cv.detail.ImageFeatures.
    # Compute them with ORB.

    # Initiate ORB detector
    orb = cv.ORB_create(
        nfeatures=keypoints.__len__() + 1,  # Do not limit amount of keypoints
        edgeThreshold=0  # Do not skip stars near the edges
    )

    if "Reduce number of keypoints":
        print("Measuring star brightnesses.")
        kp_brightnesses = [(kp, get_star_brightness(kp, orig_image)) for kp in keypoints]
        print("Measuring star brightnesses done.")

        # Keep the n brightest stars
        print(f"Keeping the {n_brightest_stars} of {kp_brightnesses.__len__()} stars.")
        brightest_kps_only = [kp for kp, brightness in
                              sorted(kp_brightnesses, key=lambda x: x[1])[::-1][:n_brightest_stars]]

    keypoints_recomputed, descriptors = orb.compute(orig_image, brightest_kps_only)
    orb_computed_keypoints_img = cv.drawKeypoints(orig_image.copy(), keypoints_recomputed, None, color=(0, 255, 0),
                                                  flags=0)
    cv.imwrite(os.path.join(output_dir, f"{img_name}___07_orb_computed_{brightest_kps_only.__len__()}_keypoints.jpg"), orb_computed_keypoints_img)

    # cv.imshow("orb", img2)
    # cv.waitKey(0)

    if "Assemble cv.detail.ImageFeatures object":
        img_features_example = "< cv.detail.ImageFeatures 0x7f51ee73a430>"
        img_features_example = {
            "descriptors": "< cv.UMat 0x7f51c23b7b50>",  # ndarray (200,32)
            "img_idx": -577474960,
            "img_size": (1342, 894),
            "keypoints": (
                # 200 elements
                "< cv.KeyPoint 0x7f51bb43bf60>",
                "< cv.KeyPoint 0x7f51bb43bea0>"
            ),
        }

        orb_dummy = cv.ORB_create()
        # dummy
        img_features_dummy = cv.detail.computeImageFeatures2(
            orb_dummy,
            # orig_image,
            np.zeros(orig_image.shape, dtype=np.uint8),

        )
        img_features_dummy.descriptors = cv.UMat(descriptors)
        img_features_dummy.img_idx = -1  # What is this?
        img_features_dummy.img_size = orig_image.shape[:2][::-1]
        img_features_dummy.keypoints = keypoints_recomputed
        print(
            f"Returning {keypoints_recomputed.__len__()} keypoints (the {n_brightest_stars} brightest keypoints).")

    print(f"Detecting stars done for {img_name}.")
    return img_features_dummy


def get_star_brightness(keypoint, workscale_image):
    """
    Get the 'brightness' for a given star (=keypoint).
    Extract pixels inside r=10px around the keypoint
    Sum up the 10 brightes pixels.

    Parameters
    ----------
    keypoint : cv2.KeyPoint

    workscale_image : np.ndarray(shape=(:,:,3), dtype=uint8)
        3-channel image.

    Returns
    -------
    brightness : int
    	The keypoints 'brightness'.
    """

    img_grayscale = cv.cvtColor(workscale_image, cv.COLOR_BGR2GRAY)

    # Create a mask:
    height, width = img_grayscale.shape
    mask = np.zeros((height, width), np.uint8)

    # Draw the circles on that mask (set thickness to -1 to fill the circle):
    circle_img = cv.circle(mask, tuple(int(itm) for itm in keypoint.pt), int(10), (255, 255, 255), thickness=-1)

    # Copy that image using that mask:
    masked_data = cv.bitwise_and(img_grayscale, img_grayscale, mask=circle_img)

    brightness = int(np.sum(np.sort(masked_data.flatten())[-10:]))
    return brightness

# Joachim Broser 2022