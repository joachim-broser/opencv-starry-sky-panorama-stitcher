#!/bin/python3.10

import colorsys
import copy
import datetime
import json
import os
import pickle
import statistics
import traceback
from collections import defaultdict

import cv2 as cv
import math
import numpy as np
import time
from PIL import Image

import cv2_pickleable.detail
from custom_bruteforce_matcher import CustomBruteForceMatcher
from image_processors import optimize_img_for_feature_detection, compute_star_features, adjust_black_and_white_point


class Config:
    def __init__(self):
        self.input_dir = "folder_with_src_images"
        self.output_dir = "destination_folder"
        self.enforced_matches = []
        self.img_names = []
        self.try_cuda = False
        self.work_megapix = 0.6
        self.feature_detector = (cv.ORB.create(), "orb")
        self.matcher = "homography"
        self.estimator = cv.detail_HomographyBasedEstimator
        self.match_conf = None
        self.conf_thresh = 1.0
        self.ba = cv.detail_BundleAdjusterRay
        self.ba_refine_mask = 'xxxxx'
        self.wave_correct = None
        self.save_graph = None
        self.warp = "fisheye"
        self.seam_megapix = 0.1
        self.seam = (cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO), "no")
        self.compose_megapix = -1
        self.expos_comp = cv.detail.ExposureCompensator_NO
        self.expos_comp_nr_feeds = np.int32(1)
        self.expos_comp_nr_filtering = np.int32(2)
        self.expos_comp_block_size = 32
        self.blend = "no"
        self.blend_strength = np.int32(5)
        self.result_filename = "result"
        self.timelapse = "none"
        self.rangewidth = -1
        self.gif_megapix = 2
        self.black_and_white_point_adjustment = {
            "final_panorama": None
        }
        self.disable_all_prompts = False
        self.colorize_edges = False
        self.colorize_seams = False
        self.mirror_pano = False
        self.rotate_pano_rad = 0

        self.enforce_star_polygon_matcher = False
        self.predefined_overlaps = []
        self.focal_length_pinhole = None
        self.disable_star_feature_finder = True

    def set_defaults(self):
        self.input_dir = "folder_with_src_images"
        self.output_dir = "destination_folder"

        self.result_filename = "DEFAULT_FiLENAME"

        # Files to stitch
        self.img_names = []

        # Matches will be considered valid, no matter what their confidence says.
        # Makes sense, if you know (from a previous) run, that found matches are valid, although num_inliers and
        # therefore confidence is low.
        self.enforced_matches = []

        # When bruteforcing matches, the star polygon matcher could be triggered if ORB matching does detect
        # no or only poor matches. In order to avoid that any image combination is processed by the star polygon matcher
        # only the overlapping images defined here will be processed by star polygon matcher, if ORB matching fails.
        self.predefined_overlaps = []

        # Focal length of the pinhole camera.
        # Essential for calculation of spherical triangles properties.
        # Can be obtained automatically by stitching 2 daylight images.
        self.focal_length_pinhole = None

        # Sometimes, cameras get flipped. A small-planet panorama then is yielded instead of a fisheye.
        # Cameras can be reversed in 1, 2 or all 3 directions.
        self.mirror_pano = {
            k: k for k in (
                None,
                "x",
                "y",
                "z",
                "x,y",
                "x,z",
                "y,z",
                "x,y,z",
            )}[None]

        # Rotate fisheye panorama.
        self.rotate_pano_rad = -math.pi / 4  # –45°

        # Star polygon matcher will be triggerd for any image combination,
        # no matter how good the ORB matching results were.
        self.enforce_star_polygon_matcher = False

        # Try to use CUDA. The default value is no. All default values are for CPU mode.
        self.try_cuda = False

        # Resolution for image registration step. The default is 0.6 Mpx
        self.work_megapix = 0.6

        # Type of feature detector used for images matching.
        self.feature_detector = {title: (obj, title) for title, obj in [
            # 'surf': cv.xfeatures2d_SURF.create, # Not included in PIP version of OpenCV; OpenCV must be compiled locally to get this.
            ('orb', cv.ORB.create()),
            ('orb-for-starry-sky', cv.ORB.create(
                nfeatures=1000,  # maximum number of features to be retained (by default 500)
                edgeThreshold=10,
                # This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
                patchSize=30,  # default = 31
                # WTA_K=4 # WTA_K decides number of points that produce each element of the oriented BRIEF descriptor. By default it is two, ie selects two points at a time. In that case, for matching, NORM_HAMMING distance is used. If WTA_K is 3 or 4, which takes 3 or 4 points to produce BRIEF descriptor, then matching distance is defined by NORM_HAMMING2.
            )),
            ('sift', cv.SIFT_create()),
            ('brisk', cv.BRISK_create()),
            ('akaze', cv.AKAZE_create()),
        ]}["orb-for-starry-sky"]

        # Confidence for pre-filtering inside the feature matching step.
        #
        # Matches confidence threshold only for the default matchers:
        #  - BestOf2NearestMatcher
        #  - AffineBestOf2NearestMatcher
        #  - BestOf2NearestRangeMatcher
        # Inside the matcher, a match is evaluated like this:
        #   m0.distance < (1.f - match_conf_) * m1.distance
        # If this criterion is violated, the match is dropped.
        # From the remaining matches, num_inliers and total_matches will be computed,
        # together with the final confidence of the match.
        #
        # This final confidence is then compared to self.config.conf_thresh by cv.detail.leaveBiggestComponent() and
        # only matches with a greater confidence than self.config.conf_thresh will reach the bundle adjuster.
        #
        # [iMPORTANT] self.config.match_conf *must not* be confused with self.config.conf_thresh !
        # While self.config.match_conf is responsible for some kind of pre-filtering inside the matcher,
        # self.config.conf_thresh is the final confidence of the match.
        self.match_conf = None # The default is 0.65 for surf and 0.3 for orb.

        # Disable star detection with Canny Edge, useful for daylight images (when using CustomBruteForceMatcher).
        self.disable_star_feature_finder = False

        # Only matches with a better confidence than this will reach the bundle adjuster.
        # [iMPORTANT] self.config.match_conf *must not* be confused with self.config.conf_thresh !
        # While self.config.match_conf is responsible for some kind of pre-filtering inside the matcher,
        # self.config.conf_thresh is the final confidence of the match.
        self.conf_thresh = 0.8

        # Matcher used for pairwise image matching. The default is 'homography'.
        """
        Homography model is useful for creating photo panoramas captured by camera,
        while affine-based model can be used to stitch scans and object captured by
        specialized devices. Use @ref cv::Stitcher::create to get preconfigured pipeline for one
        of those models.
        @note
        Certain detailed settings of @ref cv::Stitcher might not make sense. Especially
        you should not mix classes implementing affine model and classes implementing
        Homography model, as they work with different transformations.
        """
        self.matcher = {k: k for k in ['homography', 'affine']}["homography"]

        # Type of estimator used for transformation estimation.
        self.estimator = {
            'homography': cv.detail_HomographyBasedEstimator,
            'affine': cv.detail_AffineBasedEstimator,
        }['homography']



        # Bundle adjustment cost function.
        self.ba = {
            "ray": cv.detail_BundleAdjusterRay,
            "reproj": cv.detail_BundleAdjusterReproj,
            "affine": cv.detail_BundleAdjusterAffinePartial,
            "no": cv.detail_NoBundleAdjuster,
        }["ray"]

        # Set refinement mask for bundle adjustment. It looks like 'x_xxx',
        #
        # where 'x' means refine respective parameter and '_' means don't refine,
        # and has the following format:<fx><skew><ppx><aspect><ppy>.
        # The default mask is 'xxxxx'.
        # If bundle adjustment doesn't support estimation of selected parameter then
        # the respective flag is ignored.
        self.ba_refine_mask = 'xxxxx'

        # Perform wave effect correction.
        # Crucial in order to get a perfect circular fisheye panorama!
        # If turned off, chances are that the final fisheye panorama
        # will be oval instead of circular and look distorted!
        self.wave_correct = {
            "horiz": cv.detail.WAVE_CORRECT_HORIZ,  # 0
            "no": None,
            "vert": cv.detail.WAVE_CORRECT_VERT,  # 1
            "auto": cv.detail.WAVE_CORRECT_AUTO,  # 2
        }["horiz"]

        # Save matches graph represented in DOT language to <file_name> file.
        self.save_graph = None

        # Warp surface type.
        self.warp = {
            k: k for k in (
                'spherical',
                'plane',
                'affine',
                'cylindrical',
                'fisheye',
                'stereographic',
                'compressedPlaneA2B1',
                'compressedPlaneA1.5B1',
                'compressedPlanePortraitA2B1',
                'compressedPlanePortraitA1.5B1',
                'paniniA2B1',
                'paniniA1.5B1',
                'paniniPortraitA2B1',
                'paniniPortraitA1.5B1',
                'mercator',
                'transverseMercator',
            )
        }["fisheye"]

        # Resolution for seam estimation step. The default is 0.1 Mpx.
        self.seam_megapix = 0.1

        # Seam estimation method.
        self.seam = {title: (obj, title) for title, obj in [
            ("dp_color", cv.detail_DpSeamFinder('COLOR')),
            ("dp_colorgrad", cv.detail_DpSeamFinder('COLOR_GRAD')),
            ("voronoi", cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)),
            ("no", cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)),
        ]
                     }["dp_colorgrad"]

        # Resolution for compositing step. Use -1 for original resolution. The default is -1
        # Determines the resolution of the stitched panorama.
        # Does not describe the resolution of the full panorama but the resolution of a single image within that panorama!
        # Fisheye panorams crash when compose_megapix > 4
        self.compose_megapix = -1

        # Exposure compensation method.
        self.expos_comp = {
            "gain_blocks": cv.detail.ExposureCompensator_GAIN_BLOCKS,
            "gain": cv.detail.ExposureCompensator_GAIN,
            "channel": cv.detail.ExposureCompensator_CHANNELS,
            "channel_blocks": cv.detail.ExposureCompensator_CHANNELS_BLOCKS,
            "no": cv.detail.ExposureCompensator_NO,
        }["gain_blocks"]

        # Number of exposure compensation feed.
        self.expos_comp_nr_feeds = np.int32(1)

        # Number of filtering iterations of the exposure compensation gains.
        self.expos_comp_nr_filtering = np.int32(2)

        # BLock size in pixels used by the exposure compensator. The default is 32.
        self.expos_comp_block_size = 32

        # Blending method.
        self.blend = {k: k for k in (
            'multiband',
            'feather',  # Simple blender which mixes images at its borders.
            'no',
        )}["multiband"]

        # Blending strength from [0,100] range. The default is 5"
        # Use a higher value to erase vignetting and dark seams.
        self.blend_strength = np.int32(5)



        # Output warped images separately as frames of a time lapse movie,
        # with 'fixed_' prepended to input file names.
        self.timelapse = {
            "as_is": "as_is",  # Same dimensiosn for timelapsed frames as for stitched panorama
            "crop": "crop",  # Crop timelapsed frame down to warped image
            "none": None  # Disable timelapsing
        }["as_is"]

        # uses range_width to limit number of images to match with.
        self.rangewidth = -1

        # Resolution for the animated GIF. Resolution of the full panorama, not the single image.
        self.gif_megapix = 2 * 2  # 2.000 × 2.000 pixels
        self.gif_megapix = 2  # 1414 × 1414 pixels

        # Adjust black and white point
        # – for the final result image.
        # Find optimal settings using e.g. GIMP
        self.black_and_white_point_adjustment = {
            "final_panorama": None
        }

        # Disable all prompt dialogs
        self.disable_all_prompts = False

        # Colorize edges of unseamed (rectangular) images
        self.colorize_edges = False

        # Colorize stitching seams
        self.colorize_seams = False



    @staticmethod
    def json_serialize_unknown_types(obj):
        """
        Serializer called for unknown types.
        """
        if not any([
            isinstance(obj, tp) for tp in [float, bool, dict, list, tuple, int, str]
        ]):
            try:
                return obj.__str__()
            except:
                return obj.__name__

        raise ValueError()

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=self.__class__.json_serialize_unknown_types)

    @property
    def dir_star_detection(self):
        dir_pth = "{}/{}_{}_01_star_detection".format(
            self.output_dir,
            self.timestamp_main,
            self.result_filename,

        )
        os.makedirs(dir_pth, exist_ok=True)
        return dir_pth

    @property
    def dir_features_keypoints(self):
        dir_pth = "{}/{}_{}_02_features_keypoints".format(
            self.output_dir,
            self.timestamp_main,
            self.result_filename,
        )
        os.makedirs(dir_pth, exist_ok=True)
        return dir_pth

    @property
    def dir_pairwise_matches(self):
        dir_pth = "{}/{}_{}_03_pairwise_matches".format(self.output_dir, self.timestamp_main, self.result_filename, )
        os.makedirs(dir_pth, exist_ok=True)
        return dir_pth

    @property
    def dir_polygon_matches(self):
        dir_pth = "{}/{}_{}_04_polygon_matches".format(self.output_dir, self.timestamp_main, self.result_filename, )
        os.makedirs(dir_pth, exist_ok=True)
        return dir_pth

    @property
    def dir_masks(self):
        dir_pth = "{}/{}_{}_05_masks_untouched".format(self.output_dir, self.timestamp_main, self.result_filename, )
        if self.colorize_edges:
            dir_pth = "{}__colored_edges".format(dir_pth)
        if self.colorize_seams:
            dir_pth = "{}__red_seams".format(dir_pth)
        os.makedirs(dir_pth, exist_ok=True)
        return dir_pth

    @property
    def dir_masks_warped_seamed(self):
        dir_pth = "{}/{}_{}_06_masks_warped_seamed".format(self.output_dir, self.timestamp_main, self.result_filename, )
        if self.colorize_edges:
            dir_pth = "{}__colored_edges".format(dir_pth)
        if self.colorize_seams:
            dir_pth = "{}__red_seams".format(dir_pth)
        os.makedirs(dir_pth, exist_ok=True)
        return dir_pth

    @property
    def dir_timelapse(self):
        dir_pth = "{}/{}_{}_07_timelapse".format(self.output_dir, self.timestamp_main, self.result_filename, )
        if self.colorize_edges:
            dir_pth = "{}__colored_edges".format(dir_pth)
        if self.colorize_seams:
            dir_pth = "{}__red_seams".format(dir_pth)
        os.makedirs(dir_pth, exist_ok=True)
        return dir_pth

    @property
    def dir_full_pano(self):
        return self.output_dir  # TODO: Change output folder
        return "{}/{}_{}_08_stitched_pano".format(self.output_dir, self.timestamp_main, self.result_filename,
                                                  self.result_filename, )

    def get_result_filename(self, animation=False, ext="jpg"):
        base_filename = "{}_{}_{}_{}-{:03d}".format(
            self.timestamp_main,
            self.result_filename,
            self.warp,
            self.blend,
            self.blend_strength)
        if self.colorize_edges:
            base_filename = "{}__colored_edges".format(base_filename)
        if self.colorize_seams:
            base_filename = "{}__red_seams".format(base_filename)
        if animation:
            base_filename = "{}__animated".format(base_filename)
            ext = "gif"

        return "{}.{}".format(base_filename, ext)


class StitchingDetailedPipeline:

    def __init__(self):
        self.config = Config()
        self.config.set_defaults()

        self.config.timestamp_main = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

        self.full_images = dict()
        self.full_img_sizes = []

        self.seam_scale_images = []

        self.seam_scale_img_subset = []
        self.seam_scale_img_names_subset = []

        self.imgs_features = []

        self.imgs_features_orb_and_stars = []
        self.imgs_features_len_orb = []

        self.work_scale_imgs = []

        self.full_img_sizes_subset = []

        self.corners = []
        self.masks_warped_and_seamed = []
        self.masks_warped_untouched = []
        self.images_warped = []
        self.images_warped_f = []
        self.sizes = []
        self.masks = []

        self.is_work_scale_set = False
        self.is_seam_scale_set = False
        self.is_compose_scale_set = False

        self.work_scale = None
        self.seam_scale = None
        self.warped_image_scale = None

        self.cameras = None
        self.timelapse_activated = None
        self.timelapse_type = None

        self.b_w_point_adjustment_prompted = False
        self.seam_work_aspect = None
        self.cameras_frozen_after_matching_and_bundle_adjusting = []
        self.pairwise_matches = None

    def pipeline_create_panorama(self):
        self.match_and_bundle_adjust()
        self.compose_imgs_to_panorama()

    def pipeline_colorize_seams_and_edges(self):
        #self.match_and_bundle_adjust()

        if "Normal panorama + timelapse + animated gif":
            self.compose_imgs_to_panorama()

        if "Colorize seams in full panorama":
            self.config.timestamp_main = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            if self.config.seam[1] == "no":
                print("Do not turn off seam estimator!")
                exit()
            # Turn blender off
            self.config.blend = "no"
            self.config.blend_strength = np.int32(0)

            self.config.colorize_seams = True
            if "Reset to state which was present after matching and bundle adjusting":
                self.corners = []
                self.masks_warped_and_seamed = []
                self.masks_warped_untouched = []
                self.images_warped = []
                self.images_warped_f = []
                self.sizes = []
                self.masks = []

                self.cameras = self.cameras_frozen_after_matching_and_bundle_adjusting
            self.compose_imgs_to_panorama()
            self.config.colorize_seams = False

        if "Do NOT apply seam masks, do colorize rectangular edges of single images: No blending":
            # Colorful edges in background images will be hidden by foreground images
            self.config.timestamp_main = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            self.config.seam = (cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO), "no")
            self.config.blend = "no"
            self.config.blend_strength = np.int32(0)

            # self.timelapse_activated = False  # No timelapse with colored edges
            self.config.colorize_edges = True
            if "Reset to state which was present after matching and bundle adjusting":
                self.corners = []
                self.masks_warped_and_seamed = []
                self.masks_warped_untouched = []
                self.images_warped = []
                self.images_warped_f = []
                self.sizes = []
                self.masks = []

                self.cameras = self.cameras_frozen_after_matching_and_bundle_adjusting
            self.compose_imgs_to_panorama()

        if "Do NOT apply seam masks, do colorize rectangular edges of single images: : Multiband blending":
            # All colorful edges will be visible
            self.config.timestamp_main = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            self.config.seam = (cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO), "no")

            self.config.blend = "multiband"
            self.config.blend_strength = np.int32(1)

            # self.timelapse_activated = False  # No timelapse with colored edges
            self.config.colorize_edges = True
            if "Reset to state which was present after matching and bundle adjusting":
                self.corners = []
                self.masks_warped_and_seamed = []
                self.masks_warped_untouched = []
                self.images_warped = []
                self.images_warped_f = []
                self.sizes = []
                self.masks = []

                self.cameras = self.cameras_frozen_after_matching_and_bundle_adjusting
            self.compose_imgs_to_panorama()

    def test_orb_patch_sizes(self):
        for patch_size in range(5, 500 + 1, 5):
            # self.config.timestamp_main = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")+"__patch_size="+str(patch_size)
            self.config.feature_detector = (cv.ORB.create(
                nfeatures=1000,  # maximum number of features to be retained (by default 500)
                edgeThreshold=0,
                # This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
                patchSize=patch_size,  # default = 31
                # WTA_K=4 # WTA_K decides number of points that produce each element of the oriented BRIEF descriptor. By default it is two, ie selects two points at a time. In that case, for matching, NORM_HAMMING distance is used. If WTA_K is 3 or 4, which takes 3 or 4 points to produce BRIEF descriptor, then matching distance is defined by NORM_HAMMING2.
            ), 'orb-for-starry-sky')
            self.match_and_bundle_adjust()
        # self.compose_imgs_to_panorama()

    def test_work_megapix(self):
        for work_pix in range(0, 5000 * 3000 + 1, int(1000 * 100 / 1))[1:]:
            self.is_work_scale_set = False
            self.config.work_megapix = work_pix / (1000 * 1000)

            # self.config.timestamp_main = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")+"__patch_size="+str(patch_size)
            self.config.feature_detector = (cv.ORB.create(
                nfeatures=100000,  # maximum number of features to be retained (by default 500)
                edgeThreshold=10,
                # This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
                patchSize=100,  # default = 31
                # WTA_K=4 # WTA_K decides number of points that produce each element of the oriented BRIEF descriptor. By default it is two, ie selects two points at a time. In that case, for matching, NORM_HAMMING distance is used. If WTA_K is 3 or 4, which takes 3 or 4 points to produce BRIEF descriptor, then matching distance is defined by NORM_HAMMING2.
            ), 'orb-for-starry-sky')
            # self.config.work_megapix = 2
            self.match_and_bundle_adjust()
        # self.compose_imgs_to_panorama()

    def test_blend_strength(self):
        self.match_and_bundle_adjust()
        for blend_width_ in range(0, 100 + 1):
            self.config.blend_strength = np.int32(blend_width_)
            self.compose_imgs_to_panorama()

    def pipeline_demonstrate_all_projections(self):
        self.config.wave_correct  # Avoid warping errors.
        self.match_and_bundle_adjust()

        for warper in (
                'spherical',
                'plane',
                'affine',
                'cylindrical',
                'fisheye',
                'stereographic',
                'compressedPlaneA2B1',
                'compressedPlaneA1.5B1',
                'compressedPlanePortraitA2B1',
                'compressedPlanePortraitA1.5B1',
                'paniniA2B1',
                'paniniA1.5B1',
                'paniniPortraitA2B1',
                'paniniPortraitA1.5B1',
                'mercator',
                'transverseMercator',
        ):
            self.config.warp = warper

            if "Reset to state after matching and bundle adjusting":
                self.corners = []
                self.masks_warped_and_seamed = []
                self.masks_warped_untouched = []
                self.images_warped = []
                self.images_warped_f = []
                self.sizes = []
                self.masks = []

                self.cameras = self.cameras_frozen_after_matching_and_bundle_adjusting

            try:
                self.compose_imgs_to_panorama()
            except Exception as e:
                traceback.print_exc()

                with open(os.path.join(self.config.output_dir, self.config.get_result_filename(ext="txt")), "w") as tf:
                    traceback.print_exc(file=tf)

    def test_blend_strength_with_colored_edges(self):
        self.match_and_bundle_adjust()
        if True:
            # Specific config for colored edges:
            self.config.seam = (cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO), "no")
            self.config.blend = "no"
            self.config.blend_strength = np.int32(0)

            self.config.blend = "multiband"
            self.config.blend_strength = np.int32(2)

            self.config.timelapse = None  # No timelapse with colored edges
            self.config.colorize_edges = True

        for blend_width_ in range(0, 100 + 1):
            self.config.blend_strength = np.int32(blend_width_)
            self.compose_imgs_to_panorama()

    def get_compensator(self):
        expos_comp_type = self.config.expos_comp
        expos_comp_nr_feeds = self.config.expos_comp_nr_feeds
        expos_comp_block_size = self.config.expos_comp_block_size
        # expos_comp_nr_filtering = self.config.expos_comp_nr_filtering
        if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
            compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv.detail_BlocksChannelsCompensator(
                expos_comp_block_size, expos_comp_block_size,
                expos_comp_nr_feeds
            )
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        else:
            compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
        return compensator

    def match_and_bundle_adjust(self):
        """
        Run the following pipeline:
        – Detect ImageFeatures (descriptors + keypoints).
        – Find pairwise matches of images.
        – Estimate transformations for each camera.
        – Bundle-adjust the found camera parameters.
        """

        print(self.config)
        if not self.config.disable_star_feature_finder:
            if self.config.focal_length_pinhole is None:
                raise ValueError("Star polygon matcher is activated but focal_length_pinhole is not set!")

        if self.config.save_graph is None:
            save_graph = False
        else:
            save_graph = True
        if self.config.timelapse is not None:
            self.timelapse_activated = True
            if self.config.timelapse == "as_is":
                self.timelapse_type = cv.detail.Timelapser_AS_IS
            elif self.config.timelapse == "crop":
                self.timelapse_type = cv.detail.Timelapser_CROP
            else:
                print("Bad timelapse method")
                exit()
        else:
            self.timelapse_activated = False

        # finder = self.config.feature_detector()

        self.seam_work_aspect = 1

        image_shapes_tmp = defaultdict(int)
        for name in self.config.img_names:
            # full_img = cv.imread(cv.samples.findFile(name))
            print("Reading img: {}".format(name))
            full_img = cv.imread(os.path.join(self.config.input_dir, name))

            if "Check if all images have the same dimensions and the same orientation":
                image_shapes_tmp[full_img.shape] += 1
                most_common_image_shape = sorted(image_shapes_tmp.items(), key=lambda x: x[1])[::-1][0][0]
                if full_img.shape != most_common_image_shape:
                    if full_img.shape[:2] == most_common_image_shape[:2][::-1]:
                        # Image (often the zenith image) has the wrong orientation.
                        full_img = cv.rotate(full_img, cv.ROTATE_90_CLOCKWISE)
                        print(f"{name} was rotated by 90° clockwise.")
                    else:
                        raise NotImplementedError("Images with different shapes are NOT supported yet by StarPolygonMatcher.")



            self.full_images[name] = full_img


            if full_img is None:
                print("Cannot read image ", name)
                exit()
            self.full_img_sizes.append((full_img.shape[1], full_img.shape[0]))

            if "Final check":
                if set(self.full_img_sizes).__len__() != 1:
                    raise NotImplementedError(
                        "Images with different shapes are NOT supported yet by StarPolygonMatcher.")

            if self.config.work_megapix < 0:
                img = full_img
                self.work_scale = 1
                self.is_work_scale_set = True
            else:
                if self.is_work_scale_set is False:
                    self.work_scale = min(1.0, np.sqrt(
                        self.config.work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    self.is_work_scale_set = True

                interpolation_algorithms = {
                    "INTER_NEAREST": 0,  # nearest neighbor interpolation
                    "INTER_LINEAR": 1,  # bilinear interpolation
                    "INTER_CUBIC": 2,  # bicubic interpolation
                    "INTER_AREA": 3,
                    # resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
                    "INTER_LANCZOS4": 4,  # Lanczos interpolation over 8x8 neighborhood
                    "INTER_LINEAR_EXACT": 5,  # Bit exact bilinear interpolation
                    "INTER_NEAREST_EXACT": 6,
                    # Bit exact nearest neighbor interpolation. This will produce same results as the nearest neighbor method in PIL, scikit-image or Matlab.
                    "INTER_MAX": 7,  # mask for interpolation codes
                }
                img = cv.resize(
                    src=full_img,
                    dsize=None,
                    fx=self.work_scale,
                    fy=self.work_scale,
                    interpolation=cv.INTER_AREA  # Crucial for detecting stars
                )
                self.work_scale_imgs.append(img)
            if self.is_seam_scale_set is False:
                if self.config.seam_megapix > 0:
                    self.seam_scale = min(1.0, np.sqrt(
                        self.config.seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                else:
                    self.seam_scale = 1.0
                self.seam_work_aspect = self.seam_scale / self.work_scale
                self.is_seam_scale_set = True

            if self.config.feature_detector[1] == "orb-for-starry-sky":
                # Get ORB features
                img_features = cv.detail.computeImageFeatures2(
                    self.config.feature_detector[0],
                    optimize_img_for_feature_detection(
                        img,
                        prompt_dialog=(
                                not self.b_w_point_adjustment_prompted and not self.config.disable_all_prompts)
                    )
                )

                img_features_example = "< cv2.detail.ImageFeatures 0x7f51ee73a430>"
                img_features_example = {
                    "descriptors": "< cv2.UMat 0x7f51c23b7b50>",  # ndarray (200,32)
                    "img_idx": -577474960,
                    "img_size": (1342, 894),
                    "keypoints": (
                        # 200 elements
                        "< cv2.KeyPoint 0x7f51bb43bf60>",
                        "< cv2.KeyPoint 0x7f51bb43bea0>"
                    ),
                }

                self.b_w_point_adjustment_prompted = True  # Show dialog only once.

                if not self.config.disable_star_feature_finder:

                    # Get star features (for polygon matcher)
                    img_features_stars = compute_star_features(
                        img,
                        name,
                        self.config.dir_star_detection,
                        n_brightest_stars=1000
                    )

                    if "Combine ORB and star features.":
                        orb_dummy = cv.ORB_create()  # Workaround to get a valid cv2.detail.ImageFeatures object
                        img_features_merged_orb_and_stars = cv.detail.computeImageFeatures2(
                            orb_dummy,
                            # orig_image,
                            np.zeros(img.shape, dtype=np.uint8),

                        )
                        img_features_merged_orb_and_stars.descriptors = cv.vconcat(
                            [img_features.descriptors, img_features_stars.descriptors])
                        img_features_merged_orb_and_stars.img_idx = -1  # What is this?
                        img_features_merged_orb_and_stars.img_size = img_features.img_size
                        img_features_merged_orb_and_stars.keypoints = img_features.keypoints + img_features_stars.keypoints

                    self.imgs_features_orb_and_stars.append(img_features_merged_orb_and_stars)
                    self.imgs_features_len_orb.append(img_features.keypoints.__len__())
                else:
                    # Star features were disabled.
                    img_features_merged_orb_and_stars = img_features
                    self.imgs_features_orb_and_stars.append(img_features)
                    self.imgs_features_len_orb.append(img_features.keypoints.__len__())


            else:
                # Only use ORB features – NO star features
                img_features = cv.detail.computeImageFeatures2(self.config.feature_detector[0], img)

                self.imgs_features_orb_and_stars.append(img_features)
                self.imgs_features_len_orb.append(img_features.keypoints.__len__())


            if "Write most prominent keypoints to disk (ORB)":
                # Scale keypoint coordinates back to full resolution (for image write to disk):
                keypoints_full_resolution = []

                if self.config.feature_detector[1] != "orb-for-starry-sky":
                    img_features_merged_orb_and_stars = img_features

                for keypoint in sorted(img_features_merged_orb_and_stars.keypoints[:img_features.keypoints.__len__()],
                                       key=lambda x: x.response, reverse=True):
                    scaled_keypoint = cv.KeyPoint()
                    for attr in [
                        "angle",
                        "class_id",
                        "octave",
                        "pt",
                        "response",
                        "size",
                    ]:
                        setattr(scaled_keypoint, attr, getattr(keypoint, attr))
                    scaled_keypoint.pt = (keypoint.pt[0] / self.work_scale, keypoint.pt[1] / self.work_scale)
                    keypoints_full_resolution.append(scaled_keypoint)

                img_display_features = cv.drawKeypoints(
                    # cv.resize(src=full_img, dsize=None, fx=self.work_scale, fy=self.work_scale, interpolation=cv.INTER_LINEAR_EXACT),
                    # full_img,
                    optimize_img_for_feature_detection(
                        full_img,
                        prompt_dialog=False
                    ) if self.config.feature_detector[1] == "orb-for-starry-sky" else full_img,
                    # img_features.keypoints,
                    keypoints_full_resolution,
                    0,  # outImage
                    (0, 0, 255),  # Color
                    # cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                    # cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
                    # flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
                )

                cv.imwrite(
                    os.path.join(
                        self.config.dir_features_keypoints,
                        "features_ORB__{}_patchSize={:04d}__found_kps={}__wMP={:03.2f}.jpg".format(
                            name.replace(".jpg", ""),
                            self.config.feature_detector[0].getPatchSize(),
                            img_features.keypoints.__len__(),
                            self.config.work_megapix
                        )
                    ),
                    img_display_features
                )

            if "Write most prominent keypoints to disk (stars)":
                if self.config.feature_detector[1] == "orb-for-starry-sky" and not self.config.disable_star_feature_finder:
                    # Scale keypoint coordinates back to full resolution (for image write to disk):
                    keypoints_full_resolution = []
                    for keypoint in sorted(img_features_merged_orb_and_stars.keypoints[img_features.keypoints.__len__():],
                                           key=lambda x: x.response, reverse=True):
                        scaled_keypoint = cv.KeyPoint()
                        for attr in [
                            "angle",
                            "class_id",
                            "octave",
                            "pt",
                            "response",
                            "size",
                        ]:
                            setattr(scaled_keypoint, attr, getattr(keypoint, attr))
                        scaled_keypoint.pt = (keypoint.pt[0] / self.work_scale, keypoint.pt[1] / self.work_scale)
                        # scaled_keypoint.response=50
                        scaled_keypoint.size = math.ceil(img.shape[
                                                             1] * 0.0745 / 10) * 10  # In 'drawKeypoints' this value will be the plotted diameter of a keypoint in pixels.
                        keypoints_full_resolution.append(scaled_keypoint)

                    img_display_features = cv.drawKeypoints(
                        # cv.resize(src=full_img, dsize=None, fx=self.work_scale, fy=self.work_scale, interpolation=cv.INTER_LINEAR_EXACT),
                        # full_img,
                        optimize_img_for_feature_detection(
                            full_img,
                            prompt_dialog=False
                        ),
                        # img_features.keypoints,
                        keypoints_full_resolution,
                        0,  # outImage
                        (0, 0, 255),  # Color
                        # cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        # cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
                        # flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
                    )

                    cv.imwrite(
                        os.path.join(
                            self.config.dir_features_keypoints,
                            "features_stars__{}___found_kps={}__wMP={:03.2f}.jpg".format(
                                name.replace(".jpg", ""),
                                # self.config.feature_detector[0].getPatchSize(),
                                img_features_stars.keypoints.__len__(),
                                self.config.work_megapix
                            )
                        ),
                        img_display_features
                    )

            self.imgs_features.append(
                img_features)  # TODO: Attribute .imgs_features not needed any more, except for default matchers ?

            # img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            img = cv.resize(
                src=full_img,
                dsize=None,
                fx=self.seam_scale,
                fy=self.seam_scale,
                interpolation=cv.INTER_AREA)
            self.seam_scale_images.append(img)



        # Find Pairwise matches of images
        if self.config.feature_detector[1] == "orb-for-starry-sky":
            # Use bruteforce matcher for starry sky images
            bf_matcher = CustomBruteForceMatcher(
                imgs_features_orb_and_stars=self.imgs_features_orb_and_stars,
                imgs_features_len_orb=self.imgs_features_len_orb,
                conf_thresh=self.config.conf_thresh,
                img_names=self.config.img_names,
                enforced_matches=self.config.enforced_matches,
                work_scale_imgs=self.work_scale_imgs,
                pairwise_matches_output_dir=self.config.dir_pairwise_matches,
                black_and_white_point_adjustment_for_imshow=self.config.black_and_white_point_adjustment[
                    "final_panorama"],
                disable_prompts=self.config.disable_all_prompts,
                enforce_star_polygon_matcher=self.config.enforce_star_polygon_matcher,
                output_dir_polygon_matching=self.config.dir_polygon_matches,
                predefined_overlaps=self.config.predefined_overlaps,
                focal_length_pinhole=self.config.focal_length_pinhole,
            )
            self.pairwise_matches = bf_matcher.get_valid_pairwise_matches_that_will_pass_bundle_adjuster_wo_crashes()
        else:
            # Default matchers (do not perform well on starry sky images)
            matcher_type = self.config.matcher  # Can be: 'homography', 'affine'
            if self.config.match_conf is None:
                if self.config.feature_detector[1] == 'orb':
                    match_conf = 0.3
                else:
                    match_conf = 0.65
            else:
                match_conf = self.config.match_conf
            range_width = self.config.rangewidth
            if matcher_type == "affine":
                """
                
                """
                matcher = cv.detail_AffineBestOf2NearestMatcher(False, self.config.try_cuda, match_conf)
            elif range_width == -1:
                """
                
                """
                matcher = cv.detail_BestOf2NearestMatcher(self.config.try_cuda, match_conf)
            else:
                """
                /** @brief Features matcher similar to cv::detail::BestOf2NearestMatcher which
                finds two best matches for each feature and leaves the best one only if the
                ratio between descriptor distances is greater than the threshold match_conf.
                Unlike cv::detail::BestOf2NearestMatcher this matcher uses affine
                transformation (affine transformation estimate will be placed in matches_info).
                """
                matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, self.config.try_cuda, match_conf)

            self.pairwise_matches = matcher.apply2(
                self.imgs_features)  # TODO: Attribute .imgs_features not needed any more, except for default matchers ?
            matcher.collectGarbage()

            if "Output matches":
                print()
                print("Found pairwise matches sorted by confidence value:")
                print()
                print('{:>8}{:>8}{:>8}{:>30}{:>30}{:>8}{:>12}{:>12}{:>12}'.format(
                    "----",
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
                for matches_info in sorted(self.pairwise_matches,
                        key=lambda x: x.confidence,
                        reverse=True
                ):
                    if (matches_info.src_img_idx, matches_info.dst_img_idx) != (-1, -1):
                        print(
                            '{:>8}{:>8}{:>8}{:>30}{:>30}{:8.2f}{:>12}{:>12}{:>12.2f}'.format(
                                {True: "[omit]", False: ""}[matches_info.confidence <= self.config.conf_thresh],
                                matches_info.src_img_idx,
                                matches_info.dst_img_idx,
                                self.config.img_names[matches_info.src_img_idx],
                                self.config.img_names[matches_info.dst_img_idx],
                                matches_info.confidence,
                                matches_info.num_inliers,
                                matches_info.matches.__len__(),
                                matches_info.num_inliers / matches_info.matches.__len__() if matches_info.matches.__len__() > 0 else matches_info.num_inliers,  # ZeroDivisionError: division by zero
                            )
                        )

                print()



        if save_graph:
            with open(os.path.join(self.config.output_dir, self.config.get_result_filename(ext="matchesGraph.txt")),
                      "w") as fh:
                # Nm=1072, Ni=20, C=0.298
                fh.write("Nm: Number of matches")
                fh.write("Ni: Number of inliers")
                fh.write("C:  Confidence")
                fh.write(
                    cv.detail.matchesGraphAsString(self.config.img_names, self.pairwise_matches,
                                                   self.config.conf_thresh)
                )

        # Keep only images we are sure are from the same panorama ( with confidence higher than x):
        indices = cv.detail.leaveBiggestComponent(self.imgs_features_orb_and_stars, self.pairwise_matches,
                                                  self.config.conf_thresh)

        print("Keeping only those images, that have matches with a confidence better than {}:".format(
            self.config.conf_thresh))
        print(indices)

        for i in range(len(indices)):
            self.seam_scale_img_names_subset.append(self.config.img_names[indices[i]])
            self.seam_scale_img_subset.append(self.seam_scale_images[indices[i]])
            self.full_img_sizes_subset.append(self.full_img_sizes[indices[i]])
        num_images = len(self.seam_scale_img_names_subset)
        if num_images < 2:
            print("Need more images")
            exit()

        estimator = self.config.estimator()
        b, self.cameras = estimator.apply(self.imgs_features_orb_and_stars, self.pairwise_matches, None)
        if not b:
            print("Homography estimation failed.")
            exit()
        for cam in self.cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = self.config.ba()
        adjuster.setConfThresh(self.config.conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        if self.config.ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if self.config.ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if self.config.ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if self.config.ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if self.config.ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)

        print("Adjusting camera parameters.")
        b, self.cameras = adjuster.apply(self.imgs_features_orb_and_stars, self.pairwise_matches, self.cameras)

        if not b:
            print("Camera parameters adjusting failed.")
            exit()

        print("Adjusting camera parameters finished.")
        if "Dump CamaraParameters to disk":
            list_of_camera_params_for_disk_output = list()
            for camera_param in self.cameras:
                list_of_camera_params_for_disk_output.append(
                    {
                        "R": camera_param.R.tolist(),
                        "aspect": camera_param.aspect,
                        "focal": camera_param.focal,
                        "ppx": camera_param.ppx,
                        "ppy": camera_param.ppy,
                        "t": camera_param.t.tolist(),
                    }
                )
            pinhole_focal_lengths = [cam.focal for cam in self.cameras]
            pinhole_focal_lengths_statistics = {
                "median": statistics.median([cam.focal for cam in self.cameras]),
                "mean": statistics.mean([cam.focal for cam in self.cameras]),
                "min": min([cam.focal for cam in self.cameras]),
                "max": max([cam.focal for cam in self.cameras]),
                "stdev": statistics.stdev([cam.focal for cam in self.cameras]),

            }
            with open(os.path.join(self.config.output_dir, self.config.get_result_filename(ext="CameraParams.json")),
                      "w") as fh:
                fh.write(json.dumps(
                    [
                        "pinhole_focal_lengths_statistics:",
                        pinhole_focal_lengths_statistics,
                        "pinhole_focal_lengths:",
                        pinhole_focal_lengths,
                        "list_of_camera_params_for_disk_output:",
                        list_of_camera_params_for_disk_output
                    ],
                    indent=2
                ))

        self.serialize_current_obj_stat_and_dump_to_disk()

    def serialize_current_obj_stat_and_dump_to_disk(self):
        """
        Store current state to disk, including full images, pairwise matches, image features and so on.
        Since cv2 objects cannot be pickled, these objects are converted before pickling. These objects will be
        converted back to cv2 objects when the state is loaded from disk.
        """

        if "Dump current state of object to disk":
            deep_copy_of_self = StitchingDetailedPipeline()

            # Only pickleable properties can be deep-copied:
            pickleable_attributes = set(self.__dict__.keys()).difference(set([
                "config",
                "imgs_features",
                "imgs_features_orb_and_stars",
                "cameras",
                # "cameras_frozen_after_matching_and_bundle_adjusting",
                "pairwise_matches"
            ]))
            for attr in pickleable_attributes:
                setattr(deep_copy_of_self, attr, copy.deepcopy(getattr(self, attr)))

            if "Copy unpickleable attributes":
                # imgs_features
                deep_copy_of_self.imgs_features = [cv2_pickleable.detail.ImageFeatures(itm) for itm in
                                                   self.imgs_features]
                # imgs_features_orb_and_stars
                deep_copy_of_self.imgs_features_orb_and_stars = [cv2_pickleable.detail.ImageFeatures(itm) for itm in
                                                                 self.imgs_features_orb_and_stars]
                # cameras
                deep_copy_of_self.cameras = [cv2_pickleable.detail.CameraParams(itm) for itm in self.cameras]

                # pairwise_matches (They are not used later on in panorama stitching but since they are the most
                #   essential result of the stitching process, the are serialized and stored anyway.)
                deep_copy_of_self.pairwise_matches = [cv2_pickleable.detail.MatchesInfo(itm) for itm in
                                                      self.pairwise_matches]

                if False:
                    # cameras_frozen_after_matching_and_bundle_adjusting
                    deep_copy_of_self.cameras_frozen_after_matching_and_bundle_adjusting = [
                        cv2_pickleable.detail.CameraParams(itm) for itm in
                        self.cameras_frozen_after_matching_and_bundle_adjusting]

                # config
                if "Clone 'config' attribute":
                    # Only pickleable properties can be deep-copied:
                    pickleable_attributes_of_config_attr = set(self.config.__dict__.keys()).difference(set([
                        "ba",
                        "feature_detector",
                        "estimator",
                        "seam",
                    ]))
                    for attr in pickleable_attributes_of_config_attr:
                        # setattr(deep_copy_of_self, attr, getattr(self, attr))
                        setattr(deep_copy_of_self.config, attr, copy.deepcopy(getattr(self.config, attr)))

                    if "Copy unpickleable properties":
                        # ba
                        deep_copy_of_self.config.ba = cv2_pickleable.detail.BundleAdjusterRay()

                        # estimator
                        deep_copy_of_self.config.estimator = cv2_pickleable.detail.HomographyBasedEstimator()

                        # feature_detector
                        if isinstance(self.config.feature_detector[0], cv.ORB):
                            deep_copy_of_self.config.feature_detector = (
                                cv2_pickleable.ORB(self.config.feature_detector[0]),
                                self.config.feature_detector[1]
                            )
                        else:
                            raise NotImplementedError("Add other detectors if needed...")

                        # seam
                        deep_copy_of_self.config.seam = (
                            cv2_pickleable.detail.DpSeamFinder(kind=self.config.seam[1]),
                            self.config.seam[1]
                        )
                        # kind must be provided additionally, since it cannot be deduced from the DpSeamFinder object.

            if "Verify the deepcopied object – make sure, that the recreation of the object and the original one are identical":
                for seq_obj in [
                    deep_copy_of_self.imgs_features,
                    deep_copy_of_self.imgs_features_orb_and_stars,
                    deep_copy_of_self.cameras,
                    # deep_copy_of_self.cameras_frozen_after_matching_and_bundle_adjusting,
                    deep_copy_of_self.cameras_frozen_after_matching_and_bundle_adjusting,
                ]:
                    for itm in seq_obj:
                        itm.to_cv2()

                for un_seq_obj in [
                    deep_copy_of_self.config.ba,
                    deep_copy_of_self.config.estimator,
                    deep_copy_of_self.config.feature_detector[0],
                    # ("<cv2_pickleable.ORB object at 0x7fdfbd48b640>", 'orb-for-starry-sky')
                    deep_copy_of_self.config.seam[0],
                ]:
                    un_seq_obj.to_cv2()

            with open(os.path.join(self.config.output_dir, self.config.get_result_filename(ext='bin')), "wb") as fh:

                for k in [
                    'config',
                    'full_images',
                    'full_img_sizes',
                    'seam_scale_images',
                    'seam_scale_img_subset',
                    'seam_scale_img_names_subset',
                    'imgs_features',
                    'imgs_features_orb_and_stars',
                    'imgs_features_len_orb',
                    'work_scale_imgs',
                    'full_img_sizes_subset',
                    'corners',
                    'masks_warped_and_seamed',
                    'masks_warped_untouched',
                    'images_warped',
                    'images_warped_f',
                    'sizes',
                    'masks',
                    'is_work_scale_set',
                    'is_seam_scale_set',
                    'is_compose_scale_set',
                    'work_scale',
                    'seam_scale',
                    'warped_image_scale',
                    'cameras',
                    'timelapse_activated',
                    'timelapse_type',
                    'b_w_point_adjustment_prompted',
                    'seam_work_aspect',
                    # 'cameras_frozen_after_matching_and_bundle_adjusting'
                ]:
                    try:
                        print(f"Pickling {k}.")
                        # deepcopy uses pickle.
                        is_pickleable = copy.deepcopy(deep_copy_of_self.__dict__[k])
                    except Exception as e:
                        print(f"{k} could not be pickled.")
                        traceback.print_exc()
                        raise ValueError(f"{k} could not be pickled.")

                pickle.dump(deep_copy_of_self, fh)

                print(f"Objects state was dumped to: {fh.name}")

    @staticmethod
    def load_state_from_disk(filepath):
        with open(filepath, "rb") as fh:
            restored_obj = pickle.load(fh)

        if "Convert the non-cv2 objects back to genuine cv2 objects":
            if "Restore cv2 attributes, that have been converted to pickleable non-cv2 objects before":
                # imgs_features
                restored_obj.imgs_features = [itm.to_cv2() for itm in
                                              restored_obj.imgs_features]
                # imgs_features_orb_and_stars
                restored_obj.imgs_features_orb_and_stars = [itm.to_cv2() for itm in
                                                            restored_obj.imgs_features_orb_and_stars]
                # cameras
                restored_obj.cameras = [itm.to_cv2() for itm in restored_obj.cameras]

                # pairwise_matches
                restored_obj.pairwise_matches = [itm.to_cv2() for itm in
                                                 restored_obj.pairwise_matches]

                if False:
                    # cameras_frozen_after_matching_and_bundle_adjusting
                    restored_obj.cameras_frozen_after_matching_and_bundle_adjusting = [
                        itm.to_cv2() for itm in
                        restored_obj.cameras_frozen_after_matching_and_bundle_adjusting]

                # config
                if "Restore 'config' attributes":
                    if "Restore cv2 attributes, that have been converted to pickleable non-cv2 objects before":
                        # ba
                        restored_obj.config.ba = restored_obj.config.ba.to_cv2()

                        # estimator
                        restored_obj.config.estimator = restored_obj.config.estimator.to_cv2()

                        # feature_detector
                        restored_obj.config.feature_detector = (
                            restored_obj.config.feature_detector[0].to_cv2(),
                            restored_obj.config.feature_detector[1]
                        )

                        # seam
                        restored_obj.config.seam = (
                            restored_obj.config.seam[0].to_cv2(),
                            restored_obj.config.seam[1]
                        )

        return restored_obj

    def compose_imgs_to_panorama(self):
        """
        Pipeline steps:
            – Remove waviness effect (if desired)
            – Flip and/or rotate image (if desired)
            – Warp images at seam scale
            – Colorize edges (if desired)
            – Estimate seams
            – Colorize seams (if desired)
            – Compensate exposure (if desired)
            – Warp images at compose scale
            – Create timelapse frames of each warped image
            – Create an animated GIF out of the timelapse frames
            – Blend images to the full panorama
        """

        print(self.config)

        if "Calculate warped_image_scale":
            focals = []
            for cam in self.cameras:
                focals.append(cam.focal)
            focals.sort()
            if len(focals) % 2 == 1:
                self.warped_image_scale = focals[len(focals) // 2]
            else:
                self.warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

        if "Keep a backup of cameras":
            # These camera backups can be compared to the cameras after wave correction.
            # Helpful if the warper crashes.

            self.cameras_frozen_after_matching_and_bundle_adjusting = []
            for cam in self.cameras:
                cam_clone = cv.detail.CameraParams()
                cam_clone.R = np.copy(cam.R)
                cam_clone.aspect = cam.aspect
                cam_clone.focal = cam.focal
                cam_clone.ppx = cam.ppx
                cam_clone.ppy = cam.ppy
                cam_clone.t = np.copy(cam.t)

                self.cameras_frozen_after_matching_and_bundle_adjusting.append(cam_clone)

        if "Remove waviness effect":
            print("Removing waviness.")
            # Tries to make panorama more horizontal (or vertical).
            # Removes the wavy effect of the resultant panorama

            # rmats Camera rotation matrices
            if self.config.wave_correct is not None:
                rmats = []
                for cam in self.cameras:
                    rmats.append(np.copy(cam.R))
                rmats = cv.detail.waveCorrect(rmats, self.config.wave_correct)
                for idx, cam in enumerate(self.cameras):
                    cam.R = rmats[idx]

        if "Flip and/or rotate image":
            print("Flip/rotate panorama.")
            # Rotation must be applied *after* waveCorrect, since waveCorrect resets rotation.
            identity_mat = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )

            if self.config.rotate_pano_rad != 0:
                R_y = np.array(
                    [
                        [math.cos(self.config.rotate_pano_rad), 0, math.sin(self.config.rotate_pano_rad)],
                        [0, 1, 0],
                        [(-1) * math.sin(self.config.rotate_pano_rad), 0, math.cos(self.config.rotate_pano_rad)],

                    ]
                )

                R_x = np.array(
                    [
                        [1, 0, 0],
                        [0, math.cos(self.config.rotate_pano_rad), -math.sin(self.config.rotate_pano_rad)],
                        [0, math.sin(self.config.rotate_pano_rad), math.cos(self.config.rotate_pano_rad)],

                    ]
                )
                R_z = np.array(
                    [
                        [math.cos(self.config.rotate_pano_rad), -math.sin(self.config.rotate_pano_rad), 0],
                        [math.sin(self.config.rotate_pano_rad), math.cos(self.config.rotate_pano_rad), 0],
                        [0, 0, 1],

                    ]
                )
                M_rot = R_y
            else:
                M_rot = identity_mat

            if self.config.mirror_pano != None:
                # Reverse z direction
                M_z = np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1],
                    ]
                )

                # Reverse y direction
                M_y = np.array(
                    [
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                    ]
                )

                # Reverse x direction
                M_x = np.array(
                    [
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ]
                )

                M_x_and_y = np.array(
                    [
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                    ]
                )

                M_x_and_z = np.array(
                    [
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1],
                    ]
                )

                M_y_and_z = np.array(
                    [
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                )

                # Reverse all 3 directions
                M_x_and_y_and_z = np.array(
                    [
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                )

                M_mirror = {
                    "x": M_x,
                    "y": M_y,
                    "z": M_z,
                    "x,y": M_x_and_y,
                    "x,z": M_x_and_z,
                    "y,z": M_y_and_z,
                    "x,y,z": M_x_and_y_and_z,
                }[self.config.mirror_pano]
            else:
                M_mirror = identity_mat

            if any([
                self.config.mirror_pano,
                self.config.rotate_pano_rad != 0
            ]):
                for cam in self.cameras:
                    cam.R = np.matmul(np.linalg.inv(M_mirror), np.matmul(np.linalg.inv(M_rot), cam.R, )).astype(
                        "float32")
            else:
                print("Do nothing: Neither flip nor rotate.")

        num_images = len(self.seam_scale_img_names_subset)
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((self.seam_scale_img_subset[i].shape[0], self.seam_scale_img_subset[i].shape[1]),
                                       np.uint8))
            self.masks.append(um)

        if "Warp images at seam scale":
            print("Warp images at seam scale")
            warper = cv.PyRotationWarper(self.config.warp,
                                         self.warped_image_scale * self.seam_work_aspect)  # warper could be nullptr?
            for idx in range(0, num_images):
                print(f"Warping img {idx + 1} of {num_images}:    {self.seam_scale_img_names_subset[idx]}")
                # print(f"Shape: {self.seam_scale_img_subset[idx].shape.__str__()}")
                K = self.cameras[idx].K().astype(np.float32)
                swa = self.seam_work_aspect
                K[0, 0] *= swa
                K[0, 2] *= swa
                K[1, 1] *= swa
                K[1, 2] *= swa
                try:
                    corner, image_wp = warper.warp(
                        self.seam_scale_img_subset[idx],
                        K,
                        self.cameras[idx].R,
                        # cv.INTER_LINEAR,
                        cv.INTER_AREA,
                        # opencv/modules/core/include/opencv2/core/base.hpp
                        cv.BORDER_REFLECT
                        # Extraploates image by reflection: `fedcba|abcdefgh|hgfedcb`
                    )
                except Exception as e:
                    # traceback.print_exc()
                    """
                    Traceback (most recent call last):
                      File "stitching_detailed_enhanced.py", line 998, in match_and_adjust
                        corner, image_wp = warper.warp(
                    cv2.error: OpenCV(4.6.0) /io/opencv/modules/core/src/ocl.cpp:5953: error: (-220:Unknown error code -220) OpenCL error CL_MEM_OBJECT_ALLOCATION_FAILURE (-4) during call: clEnqueueReadBuffer(q, handle=0x562fc7109c00, CL_TRUE, 0, sz=1552990460, data=0x7f4656de6040, 0, 0, 0) in function 'map'
                    """

                    print(f"Warping with this projection is not possible: {self.config.warp}")
                    if self.config.wave_correct is not None:
                        cv_wave_correct_kinds = {
                            0: "cv.detail.WAVE_CORRECT_HORIZ",
                            1: "cv.detail.WAVE_CORRECT_VERT",
                            2: "cv.detail.WAVE_CORRECT_AUTO",
                        }
                        print(f"Waviness correction is on: {self.config.wave_correct}")
                        print("Try to turn it off.")
                        raise ValueError(
                            f"\nWaviness correction is on: {cv_wave_correct_kinds[self.config.wave_correct]}\nTry to turn it off.") from e

                self.corners.append(corner)
                self.sizes.append((image_wp.shape[1], image_wp.shape[0]))
                self.images_warped.append(image_wp)
                project_image_top_left_corner, mask_wp = warper.warp(
                    self.masks[idx],
                    K,
                    self.cameras[idx].R,
                    cv.INTER_NEAREST,
                    # opencv/modules/core/include/opencv2/core/base.hpp
                    cv.BORDER_CONSTANT  # `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
                )
                self.masks_warped_and_seamed.append(mask_wp.get())

            # images_warped_f = []
            for img in self.images_warped:
                imgf = img.astype(np.float32)
                self.images_warped_f.append(imgf)

        if self.config.colorize_edges and self.config.colorize_seams:
            self.config.colorize_seams = False
            print("Turned self.config.colorize_seams off!")
            time.sleep(5)

        # Exposure compensator
        self.compensator = self.get_compensator()
        self.compensator.feed(corners=self.corners, images=self.images_warped, masks=self.masks_warped_and_seamed)

        if "Estimate seams":
            print("Estimating seams.")
            seam_finder = self.config.seam[0]
            self.masks_warped_and_seamed = seam_finder.find(self.images_warped_f, self.corners,
                                                            self.masks_warped_and_seamed)

            if self.config.seam[1] != "no":
                print("Estimating seams done ({}).".format(self.config.seam[1]))
            else:
                print("Seam estimator = 'no'")

        compose_scale = 1

        if "Reset corners and sizes":
            # self.corners and self.sizes have already been filled by 'warping images at seam scale'.

            self.corners = []
            self.sizes = []
            # TODO: Seperate corners and sizes for the compose step should be introduces.

        if "Use separate cameras for the compose step":
            # Resetting
            # since camera parameters are multiplied by compose_work_aspect
            # every time this method is called.
            self.is_compose_scale_set = False

            cameras_compose_step = []
            for cam in self.cameras:
                cam_clone = cv.detail.CameraParams()
                cam_clone.R = np.copy(cam.R)
                cam_clone.aspect = cam.aspect
                cam_clone.focal = cam.focal
                cam_clone.ppx = cam.ppx
                cam_clone.ppy = cam.ppy
                cam_clone.t = np.copy(cam.t)

                cameras_compose_step.append(cam_clone)

        timelapse_frames = []

        if "Make sure, timelapser is enabled/disabled":
            # Useful, if a object state was loaded and timelapsing shall be toggled.
            if self.config.timelapse is not None:
                self.timelapse_activated = True
                if self.config.timelapse == "as_is":
                    self.timelapse_type = cv.detail.Timelapser_AS_IS
                elif self.config.timelapse == "crop":
                    self.timelapse_type = cv.detail.Timelapser_CROP
                else:
                    print("Bad timelapse method")
                    exit()
            else:
                self.timelapse_activated = False

        blender_OBJECT = None
        timelapser_img_OBJECT = None
        timelapser_masks_OBJECT = None
        # https://github.com/opencv/opencv/blob/4.x/samples/cpp/stitching_detailed.cpp#L725 ?
        for idx, name in enumerate(self.seam_scale_img_names_subset):
            full_img = self.full_images[self.seam_scale_img_names_subset[idx]]

            if not self.is_compose_scale_set:
                if self.config.compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(
                        self.config.compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                self.is_compose_scale_set = True
                compose_work_aspect = compose_scale / self.work_scale
                # self.warped_image_scale *= compose_work_aspect
                # Warp images at compose scale
                warper = cv.PyRotationWarper(
                    self.config.warp,
                    # self.warped_image_scale
                    self.warped_image_scale * compose_work_aspect
                )
                for i in range(0, len(self.seam_scale_img_names_subset)):
                    cameras_compose_step[i].focal *= compose_work_aspect
                    cameras_compose_step[i].ppx *= compose_work_aspect
                    cameras_compose_step[i].ppy *= compose_work_aspect
                    sz = (int(round(self.full_img_sizes_subset[i][0] * compose_scale)),
                          int(round(self.full_img_sizes_subset[i][1] * compose_scale)))
                    K = cameras_compose_step[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras_compose_step[i].R)
                    self.corners.append(roi[0:2])
                    self.sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                # img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
                img = cv.resize(
                    src=full_img,
                    dsize=None,
                    fx=compose_scale,
                    fy=compose_scale,
                    interpolation=cv.INTER_AREA
                )
            else:
                img = full_img

            img = adjust_black_and_white_point(img, self.config.black_and_white_point_adjustment["final_panorama"])

            if self.config.colorize_edges:
                bgr_color_list = []

                lst_len = self.seam_scale_img_names_subset.__len__() if self.seam_scale_img_names_subset.__len__() % 2 == 0 else self.seam_scale_img_names_subset.__len__() + 1
                for i in range(lst_len):
                    bgr_color_list.append(hsv2rgb(i * 1 / lst_len, 1, 1)[::-1])

                # [0,1,2,3,4,5,6,7,8,9] => [0, 5, 2, 7, 4, 9, 6, 1, 8, 3]
                color_list = [bgr_color_list[i] if i % 2 == 0 else (bgr_color_list * 2)[i + int(lst_len / 2) - 2] for
                              i, itm
                              in enumerate(bgr_color_list)]

                img = highlight_border(img, color_list[idx])

            _img_size = (img.shape[1], img.shape[0])
            K = cameras_compose_step[idx].K().astype(np.float32)
            # corner, image_warped = warper.warp(img, K, cameras_compose_step[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            print(f"Warping image at compose scale: {name}")
            corner, image_warped = warper.warp(
                img,
                K,
                cameras_compose_step[idx].R,
                cv.INTER_LINEAR,
                cv.BORDER_REFLECT
            )

            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            project_image_top_left_corner, mask_warped = warper.warp(
                mask,
                K,
                cameras_compose_step[idx].R,
                cv.INTER_NEAREST,
                cv.BORDER_CONSTANT)
            self.masks_warped_untouched.append(mask_warped)  # Un-seamed mask

            cv.imwrite(
                os.path.join(self.config.dir_masks,
                             "masks_{}_0_untouched_mask.jpg".format(self.seam_scale_img_names_subset[idx])),
                mask_warped
            )

            self.compensator.apply(idx, self.corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)

            # cv.imwrite("masks_" + self.seam_scale_img_names_subset[idx] + "_0_masks_warped_and_seamed.jpg", self.masks_warped_and_seamed[idx])

            # Dilate seam mask by ~ 1 px
            dilated_mask = cv.dilate(
                self.masks_warped_and_seamed[idx],
                None
                # This is the kernel we will use to perform the operation. If we do not specify, the default is a simple 3x3 matrix.
            )
            # cv.imwrite("masks_" + self.seam_scale_img_names_subset[idx] + "_1_dilated_mask.jpg", dilated_mask)

            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0,
                                  cv.INTER_LINEAR_EXACT)
            # cv.imwrite("masks_" + self.seam_scale_img_names_subset[idx] + "_2_seam_mask.jpg", seam_mask)

            # Apply seam mask on warped mask
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)

            cv.imwrite(
                os.path.join(
                    self.config.dir_masks_warped_seamed,
                    "masks_{}_3_mask_warped_and_seamed.jpg".format(self.seam_scale_img_names_subset[idx])
                ),
                mask_warped
            )

            if self.config.colorize_seams:
                # Morphological Gradient
                # It is the difference between dilation and erosion of an image.
                # The result will look like the outline of the object.
                kernel = np.ones((10, 10), np.uint8)
                seam_mask_outline = cv.morphologyEx(mask_warped, cv.MORPH_GRADIENT, kernel)
                seam_mask_outline_inverse = cv.bitwise_not(seam_mask_outline)

                # Black out seam mask outline in image
                image_warped_s_bg = cv.bitwise_and(image_warped_s, image_warped_s, mask=seam_mask_outline_inverse)

                # Isolate seam outlines
                plain_red_img = np.zeros((image_warped_s.shape[0], image_warped_s.shape[1], 3), np.int16)
                plain_red_img[:] = (0, 0, 255)  # BGR
                seam_outline_fg = cv.bitwise_and(plain_red_img, plain_red_img, mask=seam_mask_outline)

                # Place seam outline on unmasked image
                image_warped_s = cv.add(image_warped_s_bg.get().astype(np.int16), seam_outline_fg.get())
                if False:
                    cv.imshow('res', image_warped_s.astype(np.uint8))
                    cv.waitKey(0)
                    cv.destroyAllWindows()

            if blender_OBJECT is None:  # and not self.timelapse_activated:
                blender_OBJECT = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=self.corners, sizes=self.sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.config.blend_strength / 100
                if blend_width < 1:
                    blender_OBJECT = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif self.config.blend == "multiband":
                    # Blender which uses multi-band blending algorithm (see @cite BA83).
                    blender_OBJECT = cv.detail_MultiBandBlender()
                    print("NumBands will be: {}".format((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32)))
                    blender_OBJECT.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
                elif self.config.blend == "feather":
                    # Simple blender which mixes images at its borders.
                    blender_OBJECT = cv.detail_FeatherBlender()
                    blender_OBJECT.setSharpness(1. / blend_width)
                blender_OBJECT.prepare(dst_sz)
            # elif timelapser_OBJECT is None and self.timelapse_activated:
            if timelapser_img_OBJECT is None and self.timelapse_activated:
                timelapser_img_OBJECT = cv.detail.Timelapser_createDefault(self.timelapse_type)
                timelapser_img_OBJECT.initialize(self.corners, self.sizes)
            if timelapser_masks_OBJECT is None and self.timelapse_activated:
                timelapser_masks_OBJECT = cv.detail.Timelapser_createDefault(self.timelapse_type)
                timelapser_masks_OBJECT.initialize(self.corners, self.sizes)
            if self.timelapse_activated:
                print("Timelapsing {}".format(self.seam_scale_img_names_subset[idx]))
                ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
                if False:
                    # By default, messy square images with reflections are drawn:
                    timelapser_img_OBJECT.process(
                        image_warped_s,
                        ma_tones,
                        self.corners[idx]
                    )
                else:
                    # Apply the mask + add to timelapser
                    timelapser_img_OBJECT.process(
                        cv.bitwise_and(image_warped_s, image_warped_s, mask=self.masks_warped_untouched[idx]),
                        ma_tones,
                        self.corners[idx]
                    )

                    # Add the mask to a different timelapser
                    timelapser_masks_OBJECT.process(
                        np.repeat(self.masks_warped_untouched[idx].astype(np.int16)[:, :, np.newaxis], 3, axis=2),
                        ma_tones,
                        self.corners[idx]
                    )

                pos_s = self.seam_scale_img_names_subset[idx].rfind("/")
                if pos_s == -1:
                    fixed_file_name = "fixed_" + self.seam_scale_img_names_subset[idx]
                else:
                    fixed_file_name = self.seam_scale_img_names_subset[idx][:pos_s + 1] + "fixed_" + \
                                      self.seam_scale_img_names_subset[idx][pos_s + 1:]

                # Write timelapsed img to disk
                cv.imwrite(
                    os.path.join(
                        self.config.dir_timelapse,
                        fixed_file_name
                    ),
                    timelapser_img_OBJECT.getDst()
                )

                if "Write transparent PNG":
                    out_img_timelapsed = timelapser_img_OBJECT.getDst()
                    out_mask_timelapsed = timelapser_masks_OBJECT.getDst()

                    cv.imwrite(
                        os.path.join(
                            self.config.dir_timelapse,
                            "transparent_" + fixed_file_name + ".png"
                        ),
                        np.concatenate((out_img_timelapsed.get(), out_mask_timelapsed.get()), axis=2)[:, :, 0:4]
                    )

                timelapse_frames.append(
                    np.copy(cv.cvtColor(timelapser_img_OBJECT.getDst().get().astype(np.uint8), cv.COLOR_BGR2RGB)))

            if not self.config.colorize_edges:
                # Use seamed mask
                blender_OBJECT.feed(cv.UMat(image_warped_s), mask_warped, self.corners[idx])
            else:
                # Use untouched mask
                blender_OBJECT.feed(cv.UMat(image_warped_s), self.masks_warped_untouched[idx], self.corners[idx])

        if self.timelapse_activated:
            # Animated timelapse GIF
            print("Creating GIF from timelapse frames.")
            scale_factor = self.config.gif_megapix * 1000 * 1000 / (
                    timelapse_frames[0].shape[0] * timelapse_frames[0].shape[1])
            scale_factor = math.sqrt(min(1, scale_factor))
            new_shape = tuple(
                map(int, (scale_factor * timelapse_frames[0].shape[1], scale_factor * timelapse_frames[0].shape[0])))

            print("GIF: Scaling image to {:.1f} % = {} px × {} px = {:.1f} Mpx".format(
                scale_factor * 100,
                *new_shape,
                new_shape[0] * new_shape[1] / 1e6
            )
            )

            im_pil = Image.fromarray(timelapse_frames[0]).resize(new_shape, resample=Image.Resampling.NEAREST)

            im_pil.save(
                os.path.join(self.config.dir_full_pano, self.config.get_result_filename(animation=True)),
                format="GIF",
                save_all=True,
                append_images=[Image.fromarray(nd_arr).resize(new_shape, resample=Image.Resampling.NEAREST) for nd_arr
                               in timelapse_frames],
                duration=500,  # ms
                loop=0  # loop forever
            )
            with open(
                    os.path.join(
                        self.config.dir_full_pano,
                        self.config.get_result_filename()
                    ) + ".txt",
                    "w"
            ) as tf:
                tf.write(self.config.__str__())

        if "Dump full pano to disk":
            result = None
            result_mask = None
            result, result_mask = blender_OBJECT.blend(result, result_mask)

            print("Imwriting full pano: {}".format(
                os.path.join(
                    self.config.dir_full_pano,
                    self.config.get_result_filename()
                )
            ))
            cv.imwrite(
                os.path.join(
                    self.config.dir_full_pano,
                    self.config.get_result_filename()
                ),
                result
            )
            with open(
                    os.path.join(
                        self.config.dir_full_pano,
                        self.config.get_result_filename()
                    ) + ".txt",
                    "w"
            ) as tf:
                tf.write(self.config.__str__())

        print("Done")


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def highlight_border(img, bgr_color_tpl):
    h = img.shape[0]
    w = img.shape[1]
    border_thickness = int(img.shape[1] * 0.005268703898840885)
    print(h)
    print(w)
    print(border_thickness)
    rectangles = [
        # (start x,y), (end x,y)
        [(0, 0), (w, border_thickness), ],  # top
        [(0, h - border_thickness), (w, h), ],  # bottom
        [(0, 0), (border_thickness, h)],  # left
        [(w - border_thickness, 0), (w, h)],  # right
    ]

    # random_bgr_color = hsv2rgb(random.randrange(0, 1000 + 1) / 1000, 1, 1)[::-1]

    for start_and_end_point in rectangles:
        img = cv.rectangle(img, *start_and_end_point, bgr_color_tpl, -1)

    return img


if __name__ == '__main__':
    pass
