import inspect

import cv2
import numpy as np

import cv2_pickleable

verbose_print = False

class ImageFeatures():
    def __init__(self, cv2_detail_ImageFeatures_obj):
        for attr in [
            # 'descriptors',
            'img_idx',
            'img_size',
            # 'keypoints'
        ]:
            setattr(self, attr, getattr(cv2_detail_ImageFeatures_obj, attr))

        self.descriptors = cv2_pickleable.UMat(cv2_detail_ImageFeatures_obj.descriptors)
        self.keypoints = tuple(cv2_pickleable.KeyPoint(itm) for itm in cv2_detail_ImageFeatures_obj.keypoints)

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        if False:
            # For unknown reason, an cv2.detail.ImageFeatures object cannot the hell be created like this.
            # Creating this object leads to:
            # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
            return_obj = cv2.detail.ImageFeatures(
                descriptors=self.descriptors.to_cv2(),
                img_idx=self.img_idx,  # What is this?
                img_size=self.img_size,
                keypoints=tuple(kp.to_cv2() for kp in self.keypoints),
            )
        else:
            # Workaround:
            orb_dummy = cv2.ORB_create()
            return_obj = cv2.detail.computeImageFeatures2(
                orb_dummy,
                np.zeros(self.img_size, dtype=np.uint8),

            )
            return_obj.descriptors = self.descriptors.to_cv2()
            return_obj.img_idx = self.img_idx  # What is this?
            return_obj.img_size = self.img_size
            return_obj.keypoints = tuple(kp.to_cv2() for kp in self.keypoints)

        if all([
            np.array_equal(return_obj.descriptors.get(), self.descriptors.to_cv2().get()),
            return_obj.img_idx == self.img_idx,
            return_obj.img_size == self.img_size,
            all([all([(getattr(zipped_itm[0], attr) == getattr(zipped_itm[1], attr)) for attr in
                      ['angle', 'class_id', 'octave', 'pt', 'response', 'size']]) for zipped_itm in
                 zip(return_obj.keypoints, self.keypoints)]),
        ]):
            return return_obj
        else:
            raise ValueError()


class CameraParams():
    def __init__(self, cv2_detail_CameraParams_obj):
        for attr in [
            # 'K',
            'R',
            'aspect', 'focal', 'ppx', 'ppy', 't'
        ]:
            setattr(self, attr, getattr(cv2_detail_CameraParams_obj, attr))

        # Additional, just for validation:
        self.K = np.copy(cv2_detail_CameraParams_obj.K())

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        if False:
            # Cannot be instantiated like this – leads to an empty instance, all properties are 0 / None.
            return_obj = cv2.detail.CameraParams(
                R=self.R,
                aspect=self.aspect,
                focal=self.focal,
                ppx=self.ppx,
                ppy=self.ppy,
                t=self.t
            )
        else:
            # Workaround:
            return_obj = cv2.detail.CameraParams()
            return_obj.R = self.R
            return_obj.aspect = self.aspect
            return_obj.focal = self.focal
            return_obj.ppx = self.ppx
            return_obj.ppy = self.ppy
            return_obj.t = self.t

        if all([
            np.array_equal(return_obj.R, self.R),
            return_obj.aspect == self.aspect,
            return_obj.focal == self.focal,
            return_obj.ppx == self.ppx,
            return_obj.ppy == self.ppy,
            np.array_equal(return_obj.t, self.t),

            np.array_equal(return_obj.K(), self.K),

        ]):
            return return_obj
        else:
            raise ValueError()


class BundleAdjusterRay():
    def __init__(self):
        pass

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        return cv2.detail.BundleAdjusterRay()


class HomographyBasedEstimator():
    def __init__(self):
        pass

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        return cv2.detail.HomographyBasedEstimator()


# < cv2_detail_DpSeamFinder 0x7f31762c3590>
class DpSeamFinder():
    def __init__(self, kind):
        # kind must be provided additionally, since it cannot be deduced from the DpSeamFinder object.
        self.kind = kind

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        SEAM_FIND_CHOICES = {
            name: (obj, name) for name, obj in
            [
                ("dp_color", cv2.detail_DpSeamFinder('COLOR')),
                ("dp_colorgrad", cv2.detail_DpSeamFinder('COLOR_GRAD')),
                ("voronoi", cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)),
                ("no", cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)),
            ]
        }
        return SEAM_FIND_CHOICES[self.kind][0]


class MatchesInfo():
    def __init__(self, cv2_detail_MatchesInfo_obj):
        all_attrs = ['getInliers', 'getMatches', 'H', 'confidence', 'dst_img_idx', 'inliers_mask', 'matches',
                     'num_inliers', 'src_img_idx']
        for attr in [
            "H",
            "confidence",
            "dst_img_idx",
            "inliers_mask",
            # "matches", # List of unpickleable types, that must be converted separately.
            "num_inliers",
            "src_img_idx",

        ]:
            setattr(self, attr, getattr(cv2_detail_MatchesInfo_obj, attr))

        self.matches = tuple(cv2_pickleable.DMatch(itm) for itm in cv2_detail_MatchesInfo_obj.matches)

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")

        if False:
            # Cannot be instantiated like this – leads to an empty instance, all properties are 0 / None.
            return_obj = cv2.detail.CameraParams(
                R=self.R,
                aspect=self.aspect,
                focal=self.focal,
                ppx=self.ppx,
                ppy=self.ppy,
                t=self.t
            )
        else:
            # Workaround:
            return_obj = cv2.detail.MatchesInfo()
            return_obj.H = self.H
            return_obj.confidence = self.confidence
            return_obj.dst_img_idx = self.dst_img_idx
            return_obj.inliers_mask = self.inliers_mask
            return_obj.matches = tuple(itm.to_cv2() for itm in self.matches)
            return_obj.num_inliers = self.num_inliers
            return_obj.src_img_idx = self.src_img_idx

        if all([
            np.array_equal(return_obj.H, self.H),

            return_obj.confidence == self.confidence,
            return_obj.dst_img_idx == self.dst_img_idx,
            np.array_equal(return_obj.inliers_mask, self.inliers_mask),
            all([all([(getattr(zipped_itm[0], attr) == getattr(zipped_itm[1], attr)) for attr in
                      ["distance", "imgIdx", "queryIdx", "trainIdx"]]) for zipped_itm in
                 zip(return_obj.matches, self.matches)]),
            return_obj.num_inliers == self.num_inliers,
            return_obj.src_img_idx == self.src_img_idx
        ]):
            return return_obj
        else:
            raise ValueError()
