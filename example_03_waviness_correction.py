#!/bin/python3.10

"""
Waviness correction
– In a first run stitch 8 starry sky images (showing the horizon) *without* waviness correction to show the
  waviness effect.
– Load the matching results from the first run and in a second run, stitch the images *with* waviness correction.

"""
import datetime

import cv2 as cv
import numpy as np

import stitching_detailed_enhanced

pano = stitching_detailed_enhanced.StitchingDetailedPipeline()

if "Adjust settings":
    pano.config.input_dir = "img_heidelstein"
    pano.config.output_dir = "example_03_waviness_correction"

    pano.config.result_filename = "no-waviness-correction"

    # Files to stitch
    pano.config.img_names = [
        "01-horiz-n.jpg",
        "02-horiz-ne.jpg",
        "03-horiz-e.jpg",
        "04-horiz-se.jpg",
        "05-horiz-s.jpg",
        "06-horiz-sw.jpg",
        "07-horiz-w.jpg",
        "08-horiz-nw.jpg",
        # "09-alt1-n.jpg",
        # "10-alt1-ne.jpg",
        # "11-alt1-e.jpg",
        # "12-alt1-se.jpg",
        # "13-alt1-s.jpg",
        # "14-alt1-sw.jpg",
        # "15-alt1-w.jpg",
        # "16-alt1-nw.jpg",
        # "17-alt2-n.jpg",
        # "18-alt2-e.jpg",
        # "19-alt2-s.jpg",
        # "20-alt2-w.jpg",
        # "21-zenith.jpg"
    ]

    # Matches will be considered valid, no matter what their confidence says.
    # Makes sense, if you know (from a previous) run, that found matches are valid, although num_inliers and
    # therefore confidence is low.
    pano.config.enforced_matches = []

    # When bruteforcing matches, the star polygon matcher could be triggered if ORB matching does detect
    # no or only poor matches. In order to avoid that any image combination is processed by the star polygon matcher
    # only the overlapping images defined here will be processed by star polygon matcher, if ORB matching fails.
    pano.config.predefined_overlaps = [
        # Horizon ↔ Horizon
        (7,0),  # NW <=> N
        (0,1),  # N  <=> NE
        (1,2),  # NE <=> E
        (2,3),  # E  <=> SE
        (3,4),  # SE <=> S
        (4,5),  # S  <=> SW
        (5,6),  # SW <=> W
        (6,7),  # W  <=> NW

        # Altitude 1 ↔ Altitude 1
        (15, 8),  # NW <=> N
        ( 8, 9),  # N  <=> NE
        ( 9,10),  # NE <=> E
        (10,11),  # E  <=> SE
        (11,12),  # SE <=> S
        (12,13),  # S  <=> SW
        (13,14),  # SW <=> W
        (14,15),  # W  <=> NW

        # Altitude 2 ↔ Altitude 2
        (16, 17),  # N <=> E
        (17, 18),  # E <=> S
        (18, 19),  # S <=> W
        (19, 16),  # W <=> N

        # Horizon ↔ Altitude 1
        (0, 8),   # N  <=> N
        (1, 9),   # NE <=> NE
        (2, 10),  # E  <=> E
        (3, 11),  # SE <=> SE
        (4, 12),  # S  <=> S
        (5, 13),  # SW <=> SW
        (6, 14),  # W  <=> W
        (7, 15),  # NW <=> NW

        # Altitude 1 ↔ Altitude 2
        (8, 16),  # N <=> N

        (10, 17),  # E <=> E

        (12, 18),  # S <=> S

        (14, 19),  # W <=> W

        # Altitude 2 ↔ Zenith
        (16, 20),  # N <=> Zenith
        (17, 20),  # E <=> Zenith
        (18, 20),  # S <=> Zenith
        (19, 20),  # W <=> Zenith

    ]

    # Focal length of the pinhole camera.
    # Essential for calculation of spherical triangles properties.
    # Can be obtained automatically by stitching 2 daylight images.
    pano.config.focal_length_pinhole = 1135

    # Sometimes, cameras get flipped. A small-planet panorama then is yielded instead of a fisheye.
    # Cameras can be reversed in 1, 2 or all 3 directions.
    pano.config.mirror_pano = {
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
    pano.config.rotate_pano_rad = 0

    # Star polygon matcher will be triggerd for any image combination,
    # no matter how good the ORB matching results were.
    pano.config.enforce_star_polygon_matcher = False

    # Try to use CUDA. The default value is no. All default values are for CPU mode.
    pano.config.try_cuda = False

    # Resolution for image registration step. The default is 0.6 Mpx
    pano.config.work_megapix = 1.2

    # Type of feature detector used for images matching.
    pano.config.feature_detector = {title: (obj, title) for title, obj in [
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
    pano.config.match_conf = None

    # Disable star detection with Canny Edge, useful for daylight images.
    pano.config.disable_star_feature_finder = False

    # Only matches with a better confidence than this will reach the bundle adjuster.
    # [iMPORTANT] self.config.match_conf *must not* be confused with self.config.conf_thresh !
    # While self.config.match_conf is responsible for some kind of pre-filtering inside the matcher,
    # self.config.conf_thresh is the final confidence of the match.
    pano.config.conf_thresh = 1.0

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
    pano.config.matcher = {k: k for k in ['homography', 'affine']}["homography"]

    # Type of estimator used for transformation estimation.
    pano.config.estimator = {
        'homography': cv.detail_HomographyBasedEstimator,
        'affine': cv.detail_AffineBasedEstimator,
    }['homography']

    # Bundle adjustment cost function.
    pano.config.ba = {
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
    pano.config.ba_refine_mask = 'xxxxx'

    # Perform wave effect correction.
    # Crucial in order to get a perfect circular fisheye panorama!
    # If turned off, chances are that the final fisheye panorama
    # will be oval instead of circular and look distorted!
    pano.config.wave_correct = {
        "horiz": cv.detail.WAVE_CORRECT_HORIZ,
        "no": None,
        "vert": cv.detail.WAVE_CORRECT_VERT,
    }["no"]

    # Save matches graph represented in DOT language to <file_name> file.
    pano.config.save_graph = None

    # Warp surface type.
    pano.config.warp = {
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
    }["cylindrical"]

    # Resolution for seam estimation step. The default is 0.1 Mpx.
    pano.config.seam_megapix = 0.1

    # Seam estimation method.
    pano.config.seam = {title: (obj, title) for title, obj in [
        ("dp_color", cv.detail_DpSeamFinder('COLOR')),
        ("dp_colorgrad", cv.detail_DpSeamFinder('COLOR_GRAD')),
        ("voronoi", cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)),
        ("no", cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)),
    ]}["dp_colorgrad"]  # good

    # Resolution for compositing step. Use -1 for original resolution. The default is -1
    # Determines the resolution of the stitched panorama.
    # Does not describe the resolution of the full panorama but the resolution of a single image within that panorama!
    # Fisheye panorams crash when compose_megapix > 4
    pano.config.compose_megapix = 0.6

    # Exposure compensation method.
    pano.config.expos_comp = {
        "gain_blocks": cv.detail.ExposureCompensator_GAIN_BLOCKS,
        "gain": cv.detail.ExposureCompensator_GAIN,
        "channel": cv.detail.ExposureCompensator_CHANNELS,
        "channel_blocks": cv.detail.ExposureCompensator_CHANNELS_BLOCKS,
        "no": cv.detail.ExposureCompensator_NO,
    }["no"]

    # Number of exposure compensation feed.
    pano.config.expos_comp_nr_feeds = np.int32(1)

    # Number of filtering iterations of the exposure compensation gains.
    pano.config.expos_comp_nr_filtering = np.int32(2)

    # BLock size in pixels used by the exposure compensator. The default is 32.
    pano.config.expos_comp_block_size = 32

    # Blending method.
    pano.config.blend = {k: k for k in (
        'multiband',
        'feather',
        'no',
    )}["multiband"]

    # Blending strength from [0,100] range. The default is 5"
    pano.config.blend_strength = np.int32(5)
    pano.config.blend_strength = np.int32(0)
    pano.config.blend_strength = np.int32(42)

    # Output warped images separately as frames of a time lapse movie,
    # with 'fixed_' prepended to input file names.
    pano.config.timelapse = {
        "as_is": "as_is",  # Same dimensiosn for timelapsed frames as for stitched panorama
        "crop": "crop",  # Crop timelapsed frame down to warped image
        "none": None  # Disable timelapsing
    }["as_is"]

    # uses range_width to limit number of images to match with.
    pano.config.rangewidth = -1

    # Adjust black and white point
    # – for the final result image.
    # Find optimal settings using e.g. GIMP
    pano.config.black_and_white_point_adjustment = {
        "final_panorama": (12, 100)
    }

    pano.config.disable_all_prompts = True

    # Colorize edges of unseamed (rectangular) images
    pano.config.colorize_edges = False

    # Colorize stitching seams
    pano.config.colorize_seams = False

if False:
    pano.pipeline_create_panorama()
else:
    pano=None

    pano_restored = stitching_detailed_enhanced.StitchingDetailedPipeline.load_state_from_disk(
        # TODO: This file cannot be provided due to GitHubs file size limitations
        # TODO: It will be created after the first run. Adjust the path!
        "example_03_waviness_correction/2022-12-30_12h33m29s_no-waviness-correction_cylindrical_multiband-042.bin"
    )

    pano_restored.config.result_filename = "horizontal-waviness-correction"
    pano_restored.config.timestamp_main = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    # Perform wave effect correction.
    # Crucial in order to get a perfect circular fisheye panorama!
    # If turned off, chances are that the final fisheye panorama
    # will be oval instead of circular and look distorted!
    pano_restored.config.wave_correct = {
        "horiz": cv.detail.WAVE_CORRECT_HORIZ,
        "no": None,
        "vert": cv.detail.WAVE_CORRECT_VERT,
    }["horiz"]


    pano_restored.compose_imgs_to_panorama()
