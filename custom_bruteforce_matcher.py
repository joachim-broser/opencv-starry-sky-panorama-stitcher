#!/bin/python3.10

import os
import traceback
from collections import defaultdict
from image_processors import adjust_black_and_white_point
from custom_bf_polygon_matcher import BFPolygonMatcher
import cv2 as cv
import itertools
import numpy as np


class CustomBruteForceMatcher():
    def __init__(
            self,
            *,
            imgs_features_orb_and_stars,
            imgs_features_len_orb,
            conf_thresh,
            img_names,
            enforced_matches=[],
            enforced_no_matches=[],
            work_scale_imgs,
            pairwise_matches_output_dir,
            black_and_white_point_adjustment_for_imshow=(0, 255),
            disable_prompts=False,
            enforce_star_polygon_matcher=False,
            output_dir_polygon_matching,
            predefined_overlaps,
            focal_length_pinhole
    ):
        self.imgs_features_orb_and_stars = imgs_features_orb_and_stars
        self.imgs_features_len_orb = imgs_features_len_orb
        self.conf_thresh = conf_thresh
        self.img_names = img_names
        self.enforced_matches = [set(itm) for itm in enforced_matches]
        self.enforced_no_matches = [set(itm) for itm in enforced_no_matches]
        self.confidence_when_enforced = 10  # If a match is enforced, its confidence is replaced by this value.
        self.work_scale_imgs = work_scale_imgs
        self.pairwise_matches_output_dir = pairwise_matches_output_dir
        self.black_and_white_point_adjustment_for_imshow = black_and_white_point_adjustment_for_imshow
        self.disable_prompts = disable_prompts
        self.output_dir_polygon_matching = output_dir_polygon_matching
        self.predefined_overlaps = [set(itm) for itm in predefined_overlaps]

        self.enforce_star_polygon_matcher = enforce_star_polygon_matcher

        # Keep track of image combinations, for which star features have been used
        # instead of ordinary features
        self.kind_of_pairwise_matches = dict()

        # Stores calculated polygon data from BFPolygonMatcher so polygons have to be calculated only one time per image.
        self.polygon_data_store = dict()
        self.fts_calculated_counter = list()

        self.focal_length_pinhole = focal_length_pinhole

    def combination_is_enforced_match(self, idx_tpl):
        """
        Regardless of the calculated confidence a match can be enforced by user input.

        Returns
        -------
            Combination is an enforced match.
        """

        return set(idx_tpl) in [set(itm) for itm in [(self.img_names.index(i1), self.img_names.index(i2)) for i1, i2 in
                                                     self.enforced_matches]]

    def combination_is_enforced_no_match(self, idx_tpl):
        """
        Regardless of the calculated confidence a match can be discarded by user input.

        Returns
        -------
            Combination is NO match.
        """

        return set(idx_tpl) in [set(itm) for itm in [(self.img_names.index(i1), self.img_names.index(i2)) for i1, i2 in
                                                     self.enforced_no_matches]]

    def get_orb_keypoint_indices_only(self):
        pass

    def get_star_keypoint_indices_only(self):
        pass

    def match_2_features(self, src_idx, dst_idx):
        """
        Find the 1 best matching relation between an ImageFeature set.
        This matching relation, a pairwise match consisting of 2 opposite cv2.detail.MatchesInfo objects
        contains all keypoint matching relations as cv2.DMatch objects.

        Returns
        -------
        pairwise_match: [
            cv2.detail.MatchesInfo({
                "H": np.array([
                    [2.26441383e+00, - 5.99990969e-01, - 1.87458247e+03],
                    [9.57051670e-01, 1.94283347e+00, - 1.00613937e+03],
                    [1.02625130e-03, - 1.93244644e-04, 1.00000000e+00]
                ]),
                "confidence": 3.0,
                "dst_img_idx": 1,
                "inliers_mask": np.array(
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                "matches": (
                    cv2.DMatch({
                        "distance": 0.10838960111141205,
                        "imgIdx": 0,
                        "queryIdx": 228,
                        "trainIdx": 253,
                    }),
                    cv2.DMatch
                ),
                "num_inliers": 16,
                "src_img_idx": 0,
            }),
            cv2.detail.MatchesInfo({
                "H": np.array([
                    [0.18927933133875316, 0.10417104023371783, 459.63030053196815],
                    [-0.21539138639114455, 0.4534087365099825, 52.42346229548164],
                    [-0.00023587139245225723, -1.928685621945817e-05, 0.5384343576986185]
                ]),
                "confidence": 3.0,
                "dst_img_idx": 0,
                "inliers_mask": np.array(
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                "matches": (
                    cv2.DMatch({
                        "distance": 0.10838960111141205,
                        "imgIdx": 0,
                        "queryIdx": 253,
                        "trainIdx": 228,
                    }),
                    cv2.DMatch
                ),
                "num_inliers": 16,
                "src_img_idx": 1,
            }),
        ]
            A pairwise match consisting of 2 opposite DMatches.
        """

        print("self.fts_calculated_counter has {} items: [{}]".format(
            self.fts_calculated_counter.__len__(),
            ", ".join([str(itm) for itm in self.fts_calculated_counter])
        ))

        ft1 = self.imgs_features_orb_and_stars[src_idx]
        ft2 = self.imgs_features_orb_and_stars[dst_idx]

        if False:
            # Select matches by Lowe criterion
            bf = cv.BFMatcher(cv.NORM_HAMMING)
            # Get k=2 best matches
            matching_keypoint_descriptors = bf.knnMatch(ft1.descriptors, ft2.descriptors, k=2)

            lowe_ratio = 0.75
            matching_keypoint_descriptor_good = []

            # Evaluate k=2 best matches
            for m, n in matching_keypoint_descriptors:
                # The lower the Distance between descriptors, the better it is.
                # The lower m.distance and the higher n.distance, the more reliable is m.
                if m.distance < lowe_ratio * n.distance:
                    matching_keypoint_descriptor_good.append(m)
            print(
                "Found {:>4} matches between [{:>3}] {:>25} and [{:>3}] {:>25}, {:>4} remaining at a lowe_ratio of {}.".format(
                    matching_keypoint_descriptors.__len__(),
                    src_idx,
                    self.img_names[src_idx][:24],
                    dst_idx,
                    self.img_names[dst_idx][:24],
                    matching_keypoint_descriptor_good.__len__(),
                    lowe_ratio
                ))
        else:
            """
            https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
            Second param is boolean variable, crossCheck which is false by default. 
            If it is true, Matcher returns only those matches with value (i,j) 
            such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. 
            That is, the two features in both sets should match each other. 
            It provides consistent result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.
            
            """
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

            matching_keypoint_descriptor_good = bf.match(
                # Only match ORB features here, discard star features
                cv.UMat(ft1.descriptors.get()[0:self.imgs_features_len_orb[src_idx], :]),
                cv.UMat(ft2.descriptors.get()[0:self.imgs_features_len_orb[dst_idx], :])
            )
            print(
                "\nFound {:>4} matches between [{:>3}] {:>25} and [{:>3}] {:>25}, {:>4} remaining due to crossCheck=True.".format(
                    matching_keypoint_descriptor_good.__len__(),
                    src_idx,
                    self.img_names[src_idx][:24],
                    dst_idx,
                    self.img_names[dst_idx][:24],
                    matching_keypoint_descriptor_good.__len__(),
                ))

        src_pts = np.float32([ft1.keypoints[m.queryIdx].pt for m in matching_keypoint_descriptor_good]).reshape(-1, 1,
                                                                                                                2)
        dst_pts = np.float32([ft2.keypoints[m.trainIdx].pt for m in matching_keypoint_descriptor_good]).reshape(-1, 1,
                                                                                                                2)

        if matching_keypoint_descriptor_good.__len__() >= 6:
            """
            When num good matches < 5:
            cv2.error: OpenCV(4.6.0) opencv-4.6.0/modules/calib3d/src/fundam.cpp:385: error: (-28:Unknown error code -28) 
            The input arrays should have at least 4 corresponding point sets to calculate Homography in function 'findHomography'
            """
            # Get the transformation matrix
            M, mask = cv.findHomography(
                src_pts,
                dst_pts,
                # Method used to compute a homography matrix. The following methods are possible:

                # - 0 - a regular method using all the points, i.e., the least squares method
                # - RANSAC - RANSAC-based robust method
                # - LMEDS - Least-Median robust method
                # - RHO - PROSAC-based robust method
                method=cv.RANSAC,
                # Maximum allowed reprojection error to treat a point pair as an inlier
                # (used in the RANSAC and RHO methods only). That is, if
                #
                # |dstPoints_i−convertPointsHomogeneous(H⋅srcPoints_i)| ** 2 > ransacReprojThreshold
                # then the point i is considered as an outlier.
                # If srcPoints and dstPoints are measured in pixels,
                # it usually makes sense to set this parameter somewhere in the range of 1 to 10.
                # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
                ransacReprojThreshold=5  # px
            )
            matchesMask = mask.ravel().tolist()
        else:
            M = None
            mask = []
            matchesMask = [0]

        pairwise_match = [
            cv.detail.MatchesInfo(),  # Original match
            cv.detail.MatchesInfo()  # Mirrored version of original match
        ]

        # Create original match
        pairwise_match[0].dst_img_idx = dst_idx
        pairwise_match[0].src_img_idx = src_idx
        pairwise_match[0].matches = tuple(matching_keypoint_descriptor_good)
        pairwise_match[0].H = M
        pairwise_match[0].inliers_mask = tuple(matchesMask)
        pairwise_match[0].num_inliers = int(sum(tuple(matchesMask)))

        # Confidence two images are from the same panorama
        # https://github.com/cdemel/OpenCV/blob/2cd76233f9b1aa30c885d6a59bc9dc8f77bddda9/modules/stitching/src/matchers.cpp
        # These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
        # using Invariant Features"
        pairwise_match[0].confidence = pairwise_match[0].num_inliers / (
                8 + 0.3 * matching_keypoint_descriptor_good.__len__())

        if "Create opposite match":
            # Create mirrored version of original match
            pairwise_match[1].dst_img_idx = src_idx  # Mirrored!
            pairwise_match[1].src_img_idx = dst_idx  # Mirrored!

            pairwise_match[1].matches = tuple()
            for itm in matching_keypoint_descriptor_good:
                # Mirror every matching keypoint descriptor
                new_matching_keypoint_descriptor = cv.DMatch()

                new_matching_keypoint_descriptor.imgIdx = 0  # train image index (Value is irrelevant!)
                new_matching_keypoint_descriptor.distance = itm.distance
                new_matching_keypoint_descriptor.queryIdx = itm.trainIdx  # Mirrored! (query descriptor index)
                new_matching_keypoint_descriptor.trainIdx = itm.queryIdx  # Mirrored! (train descriptor index)
                pairwise_match[1].matches += (new_matching_keypoint_descriptor,)

            if pairwise_match[0].H is not None:
                pairwise_match[1].H = np.linalg.inv(pairwise_match[0].H)
            else:
                # This is a very bad match anyway...
                pairwise_match[1].H = None
            pairwise_match[1].confidence = pairwise_match[0].confidence  # No change!
            pairwise_match[1].inliers_mask = tuple(matchesMask)  # No change!
            pairwise_match[1].num_inliers = int(sum(tuple(matchesMask)))  # No change!

        if "Dump first ORB result to disk":
            if pairwise_match[0].num_inliers > 0:
                self.plot_pairwise_match(
                    pairwise_match,
                    "no window title",
                    prompt_dialog=False,
                    write_to_disk=True,
                    # Found via ORB or star features
                    # kind=self.kind_of_pairwise_matches[(pairwise_match[0].src_img_idx, pairwise_match[0].dst_img_idx)]
                    kind="ORB",
                    omitted="omitted__" if (pairwise_match[0].confidence < self.conf_thresh and pairwise_match[
                        0].num_inliers < 50) else ""
                )

        if pairwise_match[0].num_inliers > 50:
            # On starry sky images: If the amount of inliers is bigger than 50,
            # the match is valid, no matter how many outliers exist:
            pairwise_match[0].confidence = self.conf_thresh * 2

        if "Apply enforced matches by increasing match's confidence":
            if all([
                self.combination_is_enforced_match((src_idx, dst_idx)),
                pairwise_match[0].confidence <= self.conf_thresh
            ]):

                print("Confidence of match [{}] {} and [{}] {} was in increased from {} to {}.".format(
                    src_idx,
                    self.img_names[src_idx],
                    dst_idx,
                    self.img_names[dst_idx],
                    pairwise_match[0].confidence,
                    self.confidence_when_enforced

                ))
                confidence_originally = pairwise_match[0].confidence
                pairwise_match[0].confidence = self.confidence_when_enforced
                pairwise_match[1].confidence = self.confidence_when_enforced
                self.plot_pairwise_match(
                    pairwise_match,
                    "Are these matches good? Enforced match: Confidence of match [{}] {} and [{}] {} was in increased from {} to {}.".format(
                        src_idx,
                        self.img_names[src_idx],
                        dst_idx,
                        self.img_names[dst_idx],
                        confidence_originally,
                        pairwise_match[0].confidence,

                    ),
                    prompt_dialog=(not self.disable_prompts),
                    write_to_disk=False,
                    omitted="omitted__" if (pairwise_match[0].confidence < self.conf_thresh) else ""
                )

            elif all([
                self.combination_is_enforced_match((src_idx, dst_idx)),
                pairwise_match[0].confidence > self.conf_thresh
            ]):
                print("Confidence of match [{}] {} and [{}] {} is {} and already high enough.".format(
                    src_idx,
                    self.img_names[src_idx],
                    dst_idx,
                    self.img_names[dst_idx],
                    pairwise_match[0].confidence,
                )
                )


        if self.enforce_star_polygon_matcher or (
                (pairwise_match[0].confidence < self.conf_thresh or pairwise_match[0].num_inliers < 12) and (
                set((src_idx, dst_idx)) in self.predefined_overlaps)):
            # ORB matching results were too bad.
            # Use custom polygon matcher for better results.
            print(
                "ORB: Confidence of this match is {:.3f} with {} inliers of {} total matches ({:.1f} %). => Use custom star polygon matcher.".format(
                    pairwise_match[0].confidence,
                    pairwise_match[0].num_inliers,
                    matching_keypoint_descriptor_good.__len__(),
                    pairwise_match[0].num_inliers / matching_keypoint_descriptor_good.__len__() * 100,
                ))
            orb_confidence_for_print = pairwise_match[0].confidence
            orb_num_inliers_for_print = pairwise_match[0].num_inliers

            # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            ft1_example = {
                "descriptors": "< cv2.UMat 0x7f38e19e9c10>",
                "descriptors.get()": "ndarray (5517,32), dtype=uint8",
                "img_idx": 997359296,
                "img_size": (1342, 894),
                "keypoints": (
                    # 5517 keypoints
                    "< cv2.KeyPoint 0x7f38e15cca20>, "
                    "< cv2.KeyPoint 0x7f38e15cf120>",
                    {
                        "angle": 79.99788665771484,
                        "class_id": -1,
                        "octave": 0,
                        "pt": (891.0, 11.0),
                        "response": 2.755453124336782e-06,
                        "size": 30.0,
                    }
                )
            }

            # matching_keypoint_descriptor_good = bf.match(ft1.descriptors, ft2.descriptors)

            # ft1 hat 5517 keypoints
            # ft2 hat 4436 keypoints

            matching_keypoint_descriptor_good_example = (
                "< cv2.DMatch 0x7f15fc951890>",
                "< cv2.DMatch 0x7f15fc953f90>",
                {
                    "distance": 61.0,
                    "imgIdx": 0,  # train image index (Value is irrelevant!)
                    "queryIdx": 37,  # max: 5513 => Keypoint index in ft1  (query descriptor index)
                    "trainIdx": 211,  # max: 4435 => Keypoint index in ft2 (train descriptor index)
                }

            )

            if not np.array_equal(self.work_scale_imgs[src_idx].shape, self.work_scale_imgs[dst_idx].shape):
                raise NotImplementedError("Images with different shapes are NOT supported yet.")

            bf = BFPolygonMatcher(
                output_dir_polygon_matching=self.output_dir_polygon_matching,
                filename_img1=self.img_names[src_idx],
                filename_img2=self.img_names[dst_idx],
                idx_img1=src_idx,
                idx_img2=dst_idx,
                img_shape=self.work_scale_imgs[src_idx].shape,
                focal_length_pinhole=self.focal_length_pinhole,
            )

            # Use different keypoints

            matching_keypoint_descriptor_good = bf.match(
                ft1.keypoints, ft2.keypoints,  # match method will filter for star features and discard orb features
                self.imgs_features_len_orb[src_idx], self.imgs_features_len_orb[dst_idx],
                self.work_scale_imgs[src_idx], self.work_scale_imgs[dst_idx],
                self.polygon_data_store,
                src_idx, dst_idx,
                self.fts_calculated_counter

            )
            src_pts = np.float32([ft1.keypoints[m.queryIdx].pt for m in matching_keypoint_descriptor_good]).reshape(-1,
                                                                                                                    1,
                                                                                                                    2)
            dst_pts = np.float32([ft2.keypoints[m.trainIdx].pt for m in matching_keypoint_descriptor_good]).reshape(-1,
                                                                                                                    1,
                                                                                                                    2)

            if matching_keypoint_descriptor_good.__len__() >= 6:
                """
                When num good matches < 5:
                cv2.error: OpenCV(4.6.0) opencv-4.6.0/modules/calib3d/src/fundam.cpp:385: error: (-28:Unknown error code -28) 
                The input arrays should have at least 4 corresponding point sets to calculate Homography in function 'findHomography'
                """
                # Get the transformation matrix
                M, mask = M, mask = cv.findHomography(
                    src_pts,
                    dst_pts,
                    # Method used to compute a homography matrix. The following methods are possible:
                    #
                    # - 0 - a regular method using all the points, i.e., the least squares method
                    # - RANSAC - RANSAC-based robust method
                    # - LMEDS - Least-Median robust method
                    # - RHO - PROSAC-based robust method
                    method=cv.RANSAC,
                    # Maximum allowed reprojection error to treat a point pair as an inlier
                    # (used in the RANSAC and RHO methods only). That is, if
                    #
                    # |dstPoints_i−convertPointsHomogeneous(H⋅srcPoints_i)| ** 2 > ransacReprojThreshold
                    # then the point i is considered as an outlier.
                    # If srcPoints and dstPoints are measured in pixels,
                    # it usually makes sense to set this parameter somewhere in the range of 1 to 10.
                    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
                    ransacReprojThreshold=5  # px
                )
                matchesMask = mask.ravel().tolist()
            else:
                M = None
                mask = []
                matchesMask = [0]

            pairwise_match = [
                cv.detail.MatchesInfo(),  # Original match
                cv.detail.MatchesInfo()  # Mirrored version of original match
            ]

            # Create original match
            pairwise_match[0].dst_img_idx = dst_idx
            pairwise_match[0].src_img_idx = src_idx
            pairwise_match[0].matches = tuple(matching_keypoint_descriptor_good)
            pairwise_match[0].H = M
            pairwise_match[0].inliers_mask = tuple(matchesMask)
            pairwise_match[0].num_inliers = int(sum(tuple(matchesMask)))

            # Confidence two images are from the same panorama
            # https://github.com/cdemel/OpenCV/blob/2cd76233f9b1aa30c885d6a59bc9dc8f77bddda9/modules/stitching/src/matchers.cpp
            # These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
            # using Invariant Features"
            pairwise_match[0].confidence = pairwise_match[0].num_inliers / (
                    8 + 0.3 * matching_keypoint_descriptor_good.__len__())

            print(f"{self.img_names[src_idx]}  <==>  {self.img_names[dst_idx]}")
            print("Custom star polygon matcher: Confidence of this match is now: {:.3f} with {} inliers.".format(
                pairwise_match[0].confidence,
                pairwise_match[0].num_inliers
            ))
            print("\tCompared to ORB: Confidence of this match was: {:.3f} with {} inliers.".format(
                orb_confidence_for_print,
                orb_num_inliers_for_print
            ))

            # 6 inliers seem to be essential for successful camera adjustment
            if pairwise_match[0].num_inliers >= 6 and pairwise_match[0].confidence > 0.01:
                # If an match found by star matcher fulfills the above criterion, it *must* be a valid match,
                # so increase its confidence value:
                pairwise_match[0].confidence = int(self.conf_thresh * 3)

            # Pairwise match was found via star features
            self.kind_of_pairwise_matches[(src_idx, dst_idx)] = "stars"

            if "Create opposite match":
                # Create mirrored version of original match
                pairwise_match[1].dst_img_idx = src_idx  # Mirrored!
                pairwise_match[1].src_img_idx = dst_idx  # Mirrored!

                pairwise_match[1].matches = tuple()
                for itm in matching_keypoint_descriptor_good:
                    # Mirror every matching keypoint descriptor
                    new_matching_keypoint_descriptor = cv.DMatch()

                    new_matching_keypoint_descriptor.imgIdx = 0  # train image index (Value is irrelevant!)
                    new_matching_keypoint_descriptor.distance = itm.distance
                    new_matching_keypoint_descriptor.queryIdx = itm.trainIdx  # Mirrored! (query descriptor index)
                    new_matching_keypoint_descriptor.trainIdx = itm.queryIdx  # Mirrored! (train descriptor index)
                    pairwise_match[1].matches += (new_matching_keypoint_descriptor,)

                if pairwise_match[0].H is not None:
                    pairwise_match[1].H = np.linalg.inv(pairwise_match[0].H)
                else:
                    # This is a very bad match anyway...
                    pairwise_match[1].H = None
                pairwise_match[1].confidence = pairwise_match[0].confidence  # No change!
                pairwise_match[1].inliers_mask = tuple(matchesMask)  # No change!
                pairwise_match[1].num_inliers = int(sum(tuple(matchesMask)))  # No change!

            if "Dump STARS result to disk":
                if pairwise_match[0].num_inliers > 0:
                    self.plot_pairwise_match(
                        pairwise_match,
                        "no window title",
                        prompt_dialog=False,
                        write_to_disk=True,
                        # Found via ORB or star features
                        # kind=self.kind_of_pairwise_matches[(pairwise_match[0].src_img_idx, pairwise_match[0].dst_img_idx)]
                        kind="STARS",
                        omitted="omitted__" if (pairwise_match[0].confidence < self.conf_thresh) else ""
                    )

        else:
            # Pairwise match was found via ORB features
            self.kind_of_pairwise_matches[(src_idx, dst_idx)] = "orb"
            if (set((src_idx, dst_idx)) in self.predefined_overlaps):
                print(
                    "ORB: Confidence of this match obtained by ORB is {:.3f} with {} inliers. => No need for star polygon matcher!".format(
                        pairwise_match[0].confidence,
                        pairwise_match[0].num_inliers
                    ))
            else:
                print(
                    "ORB: Confidence of this match is {:.3f} with {} inliers. But there is no expected overlap anyway!".format(
                        pairwise_match[0].confidence,
                        pairwise_match[0].num_inliers
                    ))

        return tuple(pairwise_match)

    def match_n_features(self):
        """
        Calculates matches for all pairwise feature combinations.
        – Sorts out low confidence matches.
        – Applies enforced matches ignoring the confidence of a match.

        Returns
        -------
        matches_list_assembeled (example for 3 features): [
            cv2.detail.MatchesInfo,  # ft0 <=> ft0 (self-refercend, empty dummy)
            cv2.detail.MatchesInfo,  # ft0 <=> ft1
            cv2.detail.MatchesInfo,  # ft0 <=> ft2

            cv2.detail.MatchesInfo,  # ft1 <=> ft0
            cv2.detail.MatchesInfo,  # ft1 <=> ft1 (self-refercend, empty dummy)
            cv2.detail.MatchesInfo,  # ft1 <=> ft2

            cv2.detail.MatchesInfo,  # ft2 <=> ft0
            cv2.detail.MatchesInfo,  # ft2 <=> ft1
            cv2.detail.MatchesInfo,  # ft2 <=> ft2 (self-refercend, empty dummy)
        ]
            Pairwise matches correctly ordered in form of a list of cv2.detail.MatchesInfo objects.
            These pairwise matches have not (yet) been verified to succeed the bundle adjuster.
        """

        all_ft_combinations = list(itertools.combinations(range(self.imgs_features_orb_and_stars.__len__()), 2))

        # Bad match dummy
        # used when images do not match and
        # for matches between and image and itself
        bad_match_dummy = cv.detail.MatchesInfo()
        bad_match_dummy.H = None
        bad_match_dummy.confidence = 0.0
        bad_match_dummy.dst_img_idx = -1
        bad_match_dummy.inliers_mask = tuple()
        bad_match_dummy.matches = tuple()
        bad_match_dummy.num_inliers = 0
        bad_match_dummy.src_img_idx = -1

        pairwise_matches_by_unord_ft_idx_untouched = dict()
        pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten = dict()

        # Calculate all matches
        for (src_idx, dst_idx) in all_ft_combinations:
            pairwise_match_result = self.match_2_features(
                src_idx,
                dst_idx,
            )
            pairwise_matches_by_unord_ft_idx_untouched[(src_idx, dst_idx)] = pairwise_match_result

        if "Apply enforced matches by increasing match's confidence":
            for ft_ids, pairwise_matches in pairwise_matches_by_unord_ft_idx_untouched.items():
                pass
                # Was moved into match_2_features...

        if "Print detected matches":
            print()
            print("Found pairwise matches sorted by confidence value:")
            print(
                "(Pairwise matches with a confidence lower than {} and not part of enforced matches have already been removed.)".format(
                    self.conf_thresh
                )
            )
            print()
            print('{:>8}{:>8}{:>8}{:>30}{:>30}{:>8}{:>12}{:>12}{:>12}'.format(
                "omit?",
                "src_idx",
                "dst_idx",
                "src_name",
                "dst_name",
                "conf",
                "num_inliers",
                "num_matches",
                "inl / total"
            )
            )
            for k, v in sorted(
                    {tuple(sorted(k)): v for k, v in pairwise_matches_by_unord_ft_idx_untouched.items()}.items(),
                    key=lambda x: x[1][0].confidence,
                    reverse=True
            ):
                if k != (-1, -1):
                    print(
                        '{:>8}{:>8}{:>8}{:>30}{:>30}{:8.2f}{:>12}{:>12}{:>12.2f}'.format(
                            {True: "[omit]", False: ""}[v[0].confidence <= self.conf_thresh],
                            k[0],
                            k[1],
                            self.img_names[k[0]][:29],
                            self.img_names[k[1]][:29],
                            v[0].confidence,
                            v[0].num_inliers,
                            v[0].matches.__len__(),
                            v[0].num_inliers / v[0].matches.__len__() if v[0].matches.__len__() > 0 else v[
                                0].num_inliers,  # ZeroDivisionError: division by zero
                        )
                    )

            print()

        # Matches-Confidence per single image
        matches_confidence_per_single_image = defaultdict(list)
        for (src_idx, dst_idx), pairwise_match in pairwise_matches_by_unord_ft_idx_untouched.items():
            matches_confidence_per_single_image[src_idx].append(pairwise_match[0].confidence)
            matches_confidence_per_single_image[dst_idx].append(pairwise_match[0].confidence)

        # Handle images with 0 matches, that will crash bundle adjuster
        matches_confidence_per_single_image_better_than_threshold = {
            k: list(filter(lambda x: x > self.conf_thresh, conf_list)) for k, conf_list in
            matches_confidence_per_single_image.items()}
        images_with_no_matches = [k for k, v in matches_confidence_per_single_image_better_than_threshold.items() if
                                  v.__len__() == 0]
        for image_idx in images_with_no_matches:
            print("With the given confidence_threshold of {} there are no matches for image [{}]: {}".format(
                self.conf_thresh,
                image_idx,
                self.img_names[image_idx]
            ))
            print(
                "Bundle adjuster will fail. You should consider removing image [{}] {} from the inputs since it cannot be integrated in the panorama.".format(
                    image_idx,
                    self.img_names[image_idx]
                )
            )
        if images_with_no_matches.__len__() > 0:
            # TODO: One could implement a solution here that removes all bad images with no matches at all.
            # TODO: But this would be a whole mess since all indices then will change.
            # So the bundle adjuster crashes now and the program has to run again (after removing bad image
            # from the inputs.
            input("Press [ENTER] to continue anyway.")

        # Check pairwise matches' confidence values.
        # Pairwise matches with a confidence value lower than confidence threshold will be considered as NO match.
        # Assemble dict with ORDERED indices
        for (src_idx, dst_idx) in all_ft_combinations:
            if all([
                not self.combination_is_enforced_no_match((src_idx, dst_idx)),
                pairwise_matches_by_unord_ft_idx_untouched[(src_idx, dst_idx)][0].confidence > self.conf_thresh,
                pairwise_matches_by_unord_ft_idx_untouched[(src_idx, dst_idx)][0].num_inliers > 5  # TODO: Change this? Impact?
            ]):
                pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten[(src_idx, dst_idx)], \
                pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten[
                    (dst_idx, src_idx)] = pairwise_matches_by_unord_ft_idx_untouched[(src_idx, dst_idx)]
            else:
                pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten[(src_idx, dst_idx)], \
                pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten[
                    (dst_idx, src_idx)] = bad_match_dummy, bad_match_dummy

        pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten[(-1, -1)] = bad_match_dummy

        if False:
            # "Balance num_inliers2":
            # ORB generates up to 400 inliers while star matcher generates up to 20 inliers
            # num_inliers is used for weighting in bundle adjuster
            # But discard this: The stitching result did not benefit from this here.
            a = 0

            for id_tpl in pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten.keys():
                matches_info = pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten[id_tpl]
                keep_n_best_DMatches = 20

                if matches_info.num_inliers > matches_info.num_inliers:

                    best_DMatches_indexed = sorted([(i, itm) for i, itm in
                                                    enumerate(matches_info.matches) if
                                                    matches_info.inliers_mask[i] == 1],
                                                   key=lambda i_and_match: i_and_match[1].distance)[
                                            :keep_n_best_DMatches]

                    indices_of_best_DMatches = [itm[0] for itm in best_DMatches_indexed]
                    inliers_mask_reduced = np.zeros(matches_info.inliers_mask.shape[0], dtype="uint8")

                    for idx in indices_of_best_DMatches:
                        inliers_mask_reduced[idx] = 1  # keep inlier

                    matches_info.inliers_mask = inliers_mask_reduced

                    matches_info.num_inliers = int(matches_info.inliers_mask.sum())
                    if matches_info.num_inliers != keep_n_best_DMatches:
                        raise ValueError(
                            f"matches_info.num_inliers {matches_info.num_inliers} != keep_n_best_DMatches {keep_n_best_DMatches}")

                    # Does this have any impact?
                    matches_info.confidence = self.conf_thresh + 1

        # The order of this list is crucial
        matches_list_assembeled = tuple(
            pairwise_matches_by_ordered_ft_idx_bad_ones_overwritten[(src_idx, dst_idx)] for (src_idx, dst_idx) in
            self.strict_order_for_pairwise_matches_list()
        )

        return matches_list_assembeled

    def strict_order_for_pairwise_matches_list(self):
        """
        Order of a list of pairwise matches between n ImageFeatures is crucial.
        If the order deviates from this, bundle adjuster will crash.

        Returns
        -------
        src_idx_dst_idx_list_order_for_pairwise_matches: list()
            A list of 2-element-tuples that represents the correct order for the given number of image features.
        """

        # For example in case of 3 features the list must look like this:
        demo_list_step_1 = [
            # src_idx, dst_idx
            (0, 0),  # self-reference
            (0, 1),
            (0, 2),

            (1, 0),
            (1, 1),  # self-reference
            (1, 2),

            (2, 0),
            (2, 1),
            (2, 2),  # self-reference
        ]

        # The self-references have to be replaced by (-1,_1):
        demo_list_final = [
            # src_idx, dst_idx
            (-1, -1),  # self-reference
            (0, 1),
            (0, 2),

            (1, 0),
            (-1, -1),  # self-reference
            (1, 2),

            (2, 0),
            (2, 1),
            (-1, -1),  # self-reference
        ]

        src_idx_dst_idx_list_order_for_pairwise_matches = []
        for i in range(self.imgs_features_orb_and_stars.__len__()):
            for j in range(self.imgs_features_orb_and_stars.__len__()):
                src_idx_dst_idx_list_order_for_pairwise_matches.append((i, j) if i != j else (-1, -1))
        return src_idx_dst_idx_list_order_for_pairwise_matches

    def get_valid_pairwise_matches_that_will_pass_bundle_adjuster_wo_crashes(self):
        """
        Find pairwise matches between all features,
        checks if bundle adjuster can be passed successfully with the found matches,
        and if not, prunes more and more matches (=replaces them by dummies)
        until bundle adjuster is passed successfully.

        Counterpart / alternative to cv.detail_BestOf2NearestMatcher.apply2 and cv.detail_BestOf2NearestRangeMatcher.apply2

        Returns
        -------
        pairwise_matches_pruned: [
            cv2.detail.MatchesInfo,
            cv2.detail.MatchesInfo,
            {
                "H": np.array([
                    [2.26441383e+00, - 5.99990969e-01, - 1.87458247e+03],
                    [9.57051670e-01, 1.94283347e+00, - 1.00613937e+03],
                    [1.02625130e-03, - 1.93244644e-04, 1.00000000e+00]
                ]),
                "confidence": 3.0,
                "dst_img_idx": 1,
                "inliers_mask": np.array(
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                "matches": (
                    cv2.DMatch,
                    cv2.DMatch,
                    {
                        "distance": 61.0,
                        "imgIdx": 0,
                        "queryIdx": 37,
                        "trainIdx": 211,
                    }
                ),
                "num_inliers": 16,
                "src_img_idx": 0,
            }

        ]
            Pairwise matches !correctly ordered! in form of a list of cv2.detail.MatchesInfo objects, that will
            pass bundle adjustment successfully.
        """

        n_pairwise_matches_of_all_features = list(self.match_n_features())
        valid_matches_indices = set(
            [tuple(sorted([itm.src_img_idx, itm.dst_img_idx])) for itm in n_pairwise_matches_of_all_features if
             all([itm.src_img_idx != -1, itm.dst_img_idx != -1])])

        # For better access to properties:
        pairwise_match_by_id = {
            (itm.src_img_idx, itm.dst_img_idx): itm for itm in n_pairwise_matches_of_all_features
        }

        # If bundle adjustment fails, there must be false positive matches.
        # Try to eliminate false positive matches.
        # Start by removing 0 matches, 1 match, 2 matches ...
        for num_of_matches_to_be_removed in range(0, valid_matches_indices.__len__()):
            print("Will try all possibilities of: Removing {} matches.".format(num_of_matches_to_be_removed))

            # Drawing (num_of_matches_to_be_removed matches) without replacement
            match_sets_to_be_tested = list(
                itertools.combinations(
                    valid_matches_indices,
                    valid_matches_indices.__len__() - num_of_matches_to_be_removed
                )
            )
            print("Will test each of the following match sets successively:")
            for itm in match_sets_to_be_tested:
                print("\t– {}".format(itm))
            print()

            match_sets_to_be_tested_and_removed_matches = [
                (itm, list(set(valid_matches_indices).difference(set(itm)))) for itm in match_sets_to_be_tested
            ]

            # Start by eliminating matches sets with the lowest sum confidence value.
            for match_ids_remaining, match_ids_removed in sorted(
                    match_sets_to_be_tested_and_removed_matches,
                    key=lambda ids_remaining_and_ids_removed: sum(
                        [pairwise_match_by_id[id].confidence for id in ids_remaining_and_ids_removed[1]]
                    )
            ):
                print("Trying bundle adjustment without matches: {}, which means: {}".format(
                    match_ids_removed,
                    match_ids_remaining.__str__()
                ))

                # Replace all matches that are not in {match_ids_remaining} by dummies
                dummy = n_pairwise_matches_of_all_features[0]
                pairwise_matches_pruned = [
                    match_ids
                    if set([match_ids.dst_img_idx, match_ids.src_img_idx]) in [set(itm) for itm in
                                                                               match_ids_remaining] else dummy
                    for match_ids in n_pairwise_matches_of_all_features
                ]

                camera_adjustment_success = self.test_camera_adjustment(self.imgs_features_orb_and_stars,
                                                                        pairwise_matches_pruned)
                if camera_adjustment_success:
                    print("Found a working camera adjustment!")
                    print("Possible bad matches:")
                    for bad_match_ids in match_ids_removed:
                        print("[{}] {}, [{}] {}: confidence = {}".format(
                            bad_match_ids[0],
                            self.img_names[bad_match_ids[0]],
                            bad_match_ids[1],
                            self.img_names[bad_match_ids[1]],
                            pairwise_match_by_id[bad_match_ids].confidence
                        ))
                    if match_ids_removed.__len__() == 0:
                        print("\nNo bad matches!")
                    print()
                    if "Statistics...":
                        print("self.fts_calculated_counter has {} items: [{}]".format(
                            self.fts_calculated_counter.__len__(),
                            ", ".join([str(itm) for itm in self.fts_calculated_counter])
                        ))
                    return pairwise_matches_pruned

            print(
                "Drew any combination of {} matches but was not successful in finding a working camera adjustment.".format(
                    num_of_matches_to_be_removed))
            print()

        raise ValueError("No success in finding a camera adjustment.")

    def test_camera_adjustment(self, features, pairwise_matches):
        """
        Verifies that the bundle adjuster can adjust this set of pairwise matches.

        :param features:
        :param pairwise_matches:
        :return: Bundle adjuster was successful.
        """

        # Rotation estimator
        # It takes features of all images, pairwise matches between all images
        # and estimates rotations of all cameras.
        #
        estimator = cv.detail_HomographyBasedEstimator()
        try:
            estimater_success, cameras = estimator.apply(features, pairwise_matches, None)
        except:
            traceback.print_exc()
            return False
            """
            Will try all possibilities of: Removing 0 matches.
            Will test each of the following match sets successively:
                    – ((0, 1), (1, 2), (2, 3), (6, 7), (4, 5), (5, 6))
            
            Trying bundle adjustment without matches: [], which means: ((0, 1), (1, 2), (2, 3), (6, 7), (4, 5), (5, 6))
            Traceback (most recent call last):
              File "/home/user/Skripte/0-DEVELOPMENT_auf_home/opencv_starry_sky_pano_stitch/example_09_18h22m_ISO1600_15s.py", line 319, in <module>
                my_pano.create_pano()
              File "/home/user/Skripte/0-DEVELOPMENT_auf_home/opencv_starry_sky_pano_stitch/stitching_detailed_enhanced.py", line 393, in create_pano
                self.match_and_adjust()
              File "/home/user/Skripte/0-DEVELOPMENT_auf_home/opencv_starry_sky_pano_stitch/stitching_detailed_enhanced.py", line 776, in match_and_adjust
                pairwise_matches = bf_matcher.get_valid_set_bund_adjustable_features()
              File "/home/user/Skripte/0-DEVELOPMENT_auf_home/opencv_starry_sky_pano_stitch/custom_bruteforce_matcher.py", line 647, in get_valid_set_bund_adjustable_features
                camera_adjustment_success = self.test_camera_adjustment(self.imgs_features_orb_and_stars, pairwise_matches_pruned)
              File "/home/user/Skripte/0-DEVELOPMENT_auf_home/opencv_starry_sky_pano_stitch/custom_bruteforce_matcher.py", line 683, in test_camera_adjustment
                estimater_success, cameras = estimator.apply(features, pairwise_matches, None)
            cv2.error: OpenCV(4.6.0) /io/opencv/modules/stitching/src/motion_estimators.cpp:1213: error: (-215:Assertion failed) centers.size() > 0 && centers.size() <= 2 in function 'findMaxSpanningTree'

            """
        if not estimater_success:
            print("Homography estimation failed.")
            exit()
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        # Camera parameters refinement algorithm, which minimizes sum of the distances
        # between the rays passing through the camera center and a feature.
        # It can estimate focal length.
        adjuster = cv.detail_BundleAdjusterRay()
        adjuster.setConfThresh(self.conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        ba_refine_mask = "xxxxx"
        if ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)

        print("Adjusting camera parameters.")
        adjuster_success, cameras = adjuster.apply(features, pairwise_matches, cameras)

        return adjuster_success

    def plot_pairwise_match(self, pairwise_match, title_text, prompt_dialog=False, write_to_disk=False, kind="",
                            omitted=""):
        """Plot / write to disk pairwise matches.
        """
        p_itm = pairwise_match[0]
        out_image_w_matches = np.empty(
            (
                max(
                    self.work_scale_imgs[p_itm.src_img_idx].shape[0],
                    self.work_scale_imgs[p_itm.dst_img_idx].shape[0]
                ),
                self.work_scale_imgs[p_itm.src_img_idx].shape[1] + self.work_scale_imgs[p_itm.dst_img_idx].shape[1], 3
            ),
            dtype=np.uint8
        )

        # Updates out_image_w_matches:
        cv.drawMatches(
            # Features wurden für full_img bestimmg.
            # img_feat
            adjust_black_and_white_point(self.work_scale_imgs[p_itm.src_img_idx],
                                         self.black_and_white_point_adjustment_for_imshow),
            self.imgs_features_orb_and_stars[p_itm.src_img_idx].keypoints,

            adjust_black_and_white_point(self.work_scale_imgs[p_itm.dst_img_idx],
                                         self.black_and_white_point_adjustment_for_imshow),
            self.imgs_features_orb_and_stars[p_itm.dst_img_idx].keypoints,

            p_itm.matches,
            out_image_w_matches,
            matchesMask=p_itm.inliers_mask,
            matchesThickness=1,
            matchColor=(255, 255, 255),
            singlePointColor=(255, 0, 0),
            # flags=cv.DrawMatchesFlags_DEFAULT #messy
            # flags=cv.DrawMatchesFlags_DRAW_OVER_OUTIMG messier
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # ok
            # flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS messiest
        )
        if write_to_disk:
            cv.imwrite(
                os.path.join(
                    self.pairwise_matches_output_dir,
                    "{}___{}___conf={:.5f}___num_inliers={}___{}___{}.jpg".format(
                        kind,
                        omitted,
                        p_itm.confidence,
                        p_itm.num_inliers,
                        self.img_names[p_itm.src_img_idx],
                        self.img_names[p_itm.dst_img_idx],
                    )
                ),
                out_image_w_matches
            )

        if all([
            # (p_itm.src_img_idx, p_itm.dst_img_idx) != (-1, -1),
            # p_itm.src_img_idx in [9]  # or p_itm.dst_img_idx in [4]
            prompt_dialog
        ]):
            cv.imshow("{}. Combining {} and {}".format(
                title_text,
                p_itm.src_img_idx,
                p_itm.dst_img_idx),
                out_image_w_matches)
            cv.waitKey()
