import inspect

import cv2

verbose_print = False

class ORB():
    def __init__(self, cv2_ORB_obj):
        self.edgeThreshold = cv2_ORB_obj.getEdgeThreshold()
        self.nfeatures = cv2_ORB_obj.getMaxFeatures()
        self.patchSize = cv2_ORB_obj.getPatchSize()
        all_cv2_ORB_properties = [
            '__new__',
            '__repr__',
            'create',
            'getDefaultName',
            'getEdgeThreshold',
            'getFastThreshold',
            'getFirstLevel',
            'getMaxFeatures',
            'getNLevels',
            'getPatchSize',
            'getScaleFactor',
            'getScoreType',
            'getWTA_K',
            'setEdgeThreshold',
            'setFastThreshold',
            'setFirstLevel',
            'setMaxFeatures',
            'setNLevels',
            'setPatchSize',
            'setScaleFactor',
            'setScoreType',
            'setWTA_K',
            '__doc__',
            '__module__',
            'compute',
            'defaultNorm',
            'descriptorSize',
            'descriptorType',
            'detect',
            'detectAndCompute',
            'empty',
            'read',
            'write',
            '__hash__',
            '__str__',
            '__getattribute__',
            '__setattr__',
            '__delattr__',
            '__lt__',
            '__le__',
            '__eq__',
            '__ne__',
            '__gt__',
            '__ge__',
            '__init__',
            '__reduce_ex__',
            '__reduce__',
            '__subclasshook__',
            '__init_subclass__',
            '__format__',
            '__sizeof__',
            '__dir__',
            '__class__'
        ]

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        return cv2.ORB.create(
            nfeatures=self.nfeatures,  # maximum number of features to be retained (by default 500)
            edgeThreshold=self.edgeThreshold,
            # This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
            patchSize=self.patchSize,  # default = 31
            # WTA_K=4 # WTA_K decides number of points that produce each element of the oriented BRIEF descriptor. By default it is two, ie selects two points at a time. In that case, for matching, NORM_HAMMING distance is used. If WTA_K is 3 or 4, which takes 3 or 4 points to produce BRIEF descriptor, then matching distance is defined by NORM_HAMMING2.
        )


class UMat():
    def __init__(self, cv2_UMat_obj):
        self.array = cv2_UMat_obj.get()

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        return cv2.UMat(self.array)


class KeyPoint():
    def __init__(self, cv2_KeyPoint_obj):
        for attr in [
            "angle",
            "class_id",
            "octave",
            "pt",
            "response",
            "size",
        ]:
            setattr(self, attr, getattr(cv2_KeyPoint_obj, attr))

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        return_obj = cv2.KeyPoint(
            x=self.pt[0],
            y=self.pt[1],
            size=self.size,
            class_id=self.class_id,
            octave=self.octave,
            response=self.response,
            angle=self.angle
        )

        if all([
            return_obj.pt == self.pt,
            return_obj.size == self.size,
            return_obj.class_id == self.class_id,
            return_obj.octave == self.octave,
            return_obj.response == self.response,
            return_obj.angle == self.angle,
        ]):
            return return_obj
        else:
            raise ValueError()


class DMatch():
    def __init__(self, cv2_DMatch_obj):
        for attr in [
            'distance', 'imgIdx', 'queryIdx', 'trainIdx'
        ]:
            setattr(self, attr, getattr(cv2_DMatch_obj, attr))

    def to_cv2(self):
        if verbose_print:
            print(f"{self.__class__.__name__}.{inspect.stack()[0][3]} called.")
        return cv2.DMatch(
            _queryIdx=self.queryIdx,
            _trainIdx=self.trainIdx,
            _distance=self.distance,
            _imgIdx=self.imgIdx
        )
