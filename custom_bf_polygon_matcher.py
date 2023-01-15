#!/bin/python3.10

import os
from collections import defaultdict

import alphashape as alphashape
import cv2 as cv
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import shapely
from sklearn.cluster import AgglomerativeClustering

from fix_descartes_library import PolygonPatch
from image_processors import optimize_img_for_feature_detection, get_star_brightness


class BFPolygonMatcher:
    """
    As an alternative to cv.BFMatcher (which matches single keypoints), BFPolygonMatcher matches star constellations /
    polygons.

    Finds matches between 2 sets of keypoints (stars) by creating polygons of stars.
    Angles, lengths of sides and star brightness of 2 polygons are compared to find the best match.

    Angles and lengths of sides are NOT measured in the image plane.
    Stars are projected to the physical world sphere. Angles and side lengths are then measured on this sphere.
    Polygon angles are angles of the spherical triangle spanned by 3 stars.
    Polygon side lengths are the side lengts of these spherical triangles and measured in radians.

    """

    def __init__(
            self,
            output_dir_polygon_matching,
            filename_img1, filename_img2,
            idx_img1, idx_img2,
            img_shape,
            focal_length_pinhole
    ):
        """

        Parameters
        ----------
        focal_length_pinhole: float
            Distance between pinhole plane and image plane in pixels. (see https://ksimek.github.io/2013/08/13/intrinsic/)
            Can be obtained by merging two daylight images.

        img_shape: np.array
            Dimensions of the image(s), needed for spherical trigonometry.
        """
        self.filename_img1 = filename_img1
        self.filename_img2 = filename_img2

        self.idx_img1 = idx_img1
        self.idx_img2 = idx_img2
        self.focal_length_pinhole = focal_length_pinhole
        self.img_shape = img_shape

        self.output_dir = os.path.join(
            output_dir_polygon_matching,
            "{}___{}".format(
                filename_img1,
                filename_img2
            )
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def dist_between_2_points(self, p1, p2):
        """ Distance in pixels between 2 points in the image plane. """
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """
        Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def angle_distance(self, p1, p2):
        """
        Calculate the angle between the 2 vectors which are spanned by the pinhole and the 2 points on the image plane.
        Image plane is focal_length_pinhole away from the pinhole.

        Parameters
        ----------
        p1 : [x_px, y_px]

        p2 : [x_px, y_px]

        Returns
        -------
        angle_v1_v2 : float (in radians)
        """
        # Shift images center from top left corner to the mid:
        half_width = int(self.img_shape[1] / 2)
        half_height = int(self.img_shape[0] / 2)

        v1 = np.array((p1[0] - half_width, p1[1] - half_height, self.focal_length_pinhole))
        v2 = np.array((p2[0] - half_width, p2[1] - half_height, self.focal_length_pinhole))
        angle_v1_v2 = self.angle_between(v1, v2)
        return angle_v1_v2

    def sphere_angle(self, p1_px, p2_px, p3_px):
        """
        Project the 3 points on the image plane through the pinhole to an sphere of arbitrary radius.
        A spherical triangle is created.
        Calculate angle of spherical triangle at vertex p2.

        Parameters
        ----------
        p1_px, p2_px, p3_px: float (in pixels)


        Returns
        -------
        angle_of_spherical_triangle : float
        """

        # p2 is the vertex of the spherical triangle, for which the angle is calculated
        angle_p1_p2 = self.angle_distance(p1_px, p2_px)  # b
        angle_p2_p3 = self.angle_distance(p2_px, p3_px)  # c
        angle_p3_p1 = self.angle_distance(p3_px, p1_px)  # a

        # Side lengths of spherical triangle are entered in radians, not in absolut lengths.

        # Spherical triangle angle at point p2 (A)
        return math.acos((math.cos(angle_p3_p1) - math.cos(angle_p1_p2) * math.cos(angle_p2_p3)) / (
                math.sin(angle_p1_p2) * math.sin(angle_p2_p3)))

    def reduce_amount_of_keypoints_by_clustering(
            self, indexed_keypoints, img_filename,
            work_scale_img,
            n_clusters=30
    ):
        """
        Reduce the amount of keypoints by sklearn.cluster.AgglomerativeClustering,
        that is removing stars that are closer to each other than a given threshold.

        Parameters
        ----------
        keypoints : {
            0: < cv2.KeyPoint 0x7ff0835c8480>,
            3: < cv2.KeyPoint 0x7ff0835c8510>,
            ...
        }
            Indexed list of keypoints to be decimated.

        Returns
        -------
        remaining_keypoints_ext_indexed : {
            52: < cv2.KeyPoint 0x7ff0835c8ea0>,
            0: < cv2.KeyPoint 0x7ff0835c8480>,
            ...
        }
            Indexed list of the remaining keypoints after clustering.
        """

        keypoints = list(indexed_keypoints.values())

        # ext_kp_ids: Keypoint ids outside this method
        # int_kp_ids: Keypoint ids inside this method (for convenience)
        ext_kp_ids_from_int_kp_ids = {i: kp_id for i, kp_id in enumerate(indexed_keypoints.keys())}

        X = np.array([kp.pt for kp in keypoints])

        keep_n_brightest_stars_per_cluster = 1

        clustering = AgglomerativeClustering(
            # distance_threshold=100, # The linkage distance threshold at or above which clusters will not be merged.
            # distance_threshold=maximum_allowed_distance_between_2_stars,
            n_clusters=n_clusters,
            # The linkage distance threshold at or above which clusters will not be merged.
            # n_clusters=None
        ).fit(X)

        unique_labels = set(list(clustering.labels_))
        labels_unique = unique_labels
        labels = clustering.labels_
        n_clusters_ = len(labels_unique)
        n_clusters = n_clusters_
        print(f"Create {n_clusters_} clusters.")

        print(f"Number of estimated clusters: {n_clusters_}")

        kp_brightnesses = np.array(tuple(get_star_brightness(kp, work_scale_img) for kp_id, kp in enumerate(keypoints)))

        if "Plot all keypoints and clusters":

            plt.figure(1)
            fig = plt.figure()
            plt.clf()

            plt.xlim([0, work_scale_img.shape[1]])
            plt.ylim([0, work_scale_img.shape[0]])
            plt.gca().invert_yaxis()

            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            markers = ["o"] * 100

            for pt_idx, pt in enumerate(X):
                plt.plot(
                    pt[0],
                    pt[1],
                    marker="o",  # markers[k],
                    markersize=(15 - 1) / (2500 - 305) * kp_brightnesses[pt_idx] + (
                            15 - 2500 * (15 - 1) / (2500 - 305)),  # kp_brightnesses[pt_idx],
                    color=colors[clustering.labels_[pt_idx]]
                )

            plt.title(f"{img_filename} Estimated number of clusters: {n_clusters_}")
            # plt.show()

            if "Debug" == "Debug":
                fig.savefig(
                    "{}/0_removed_clusters_{}__n_clusters={:04d}.jpg".format(
                        self.output_dir,
                        img_filename,
                        n_clusters
                    ), dpi=fig.dpi
                )
            plt.close("all")

        if "Keep only the 1 brightest star per cluster":
            remaining_keypoints_ext_indexed = dict()

            for cluster_id in unique_labels:
                int_kp_ids_of_this_cluster = [i for i in range(keypoints.__len__()) if labels[i] == cluster_id]

                brightnesses_of_this_cluster = [(int_kp_id, kp_brightnesses[int_kp_id]) for int_kp_id in
                                                int_kp_ids_of_this_cluster]
                # TODO: Only kick all stars but the brightest one, if the brightest one is 1.2 times brighter
                # TODO: than the 2nd brightest one!
                brightest_stars_of_cluster_int_kp_ids = [itm[0] for itm in
                                                         sorted(brightnesses_of_this_cluster, key=lambda x: x[1])[::-1][
                                                         :keep_n_brightest_stars_per_cluster]]

                for int_kp_id in brightest_stars_of_cluster_int_kp_ids:
                    remaining_keypoints_ext_indexed[ext_kp_ids_from_int_kp_ids[int_kp_id]] = keypoints[int_kp_id]

        if tuple(indexed_keypoints.keys()).__len__() != tuple(remaining_keypoints_ext_indexed.keys()).__len__():
            print("Keypoint amount dropped from {} to {} at a n_clusters = {}".format(
                tuple(indexed_keypoints.keys()).__len__(),
                tuple(remaining_keypoints_ext_indexed.keys()).__len__(),
                n_clusters
            ))
        else:
            print("Keypoint amount did not change and is still {} at a n_clusters= {}".format(
                tuple(remaining_keypoints_ext_indexed.keys()).__len__(),
                n_clusters
            ))

        return remaining_keypoints_ext_indexed

    def reduce_amount_of_keypoints_and_create_polygon_spanning_tuples_of_keypoints(
            self, keypoints, img_filename, workscale_img,
            n_polygon_sides,
            maximum_number_of_polygons_per_image_region=15000
    ):
        """
        Reduce the amount of keypoints by
           splitting the image in 4×3 = 12 image regions and making sure that each regions has at least
           n stars, reducing the number of stars in each region successively. A total number of polygons is defined
           and the number of stars in each image region is reduced simultaneously until the total number of polygons
           spanned by these stars in each image region is smaller than the specified threshold.

        and return the total amount of all possible polygons, that can be spanned by the remaining keypoints.

        Parameters
        ----------
        keypoints: tuple(cv2.KeyPoint, cv2.KeyPoint, ...)

        n_polygon_sides: int
            Number of sides of the polygons.

        maximum_number_of_polygons_per_image_region: int

        Returns
        -------
        all_possible_polygons : set([(3, 8, 56, 63, 66), (8, 45, 52, 63, 89), ...])
            All possible polygons that can be spanned by the reduced amount of keypoints in each image region.
            Polygon is defined by the keypoint indices of its vertices.
        """

        keypoints_indexed = ((i, kp) for i, kp in enumerate(keypoints))

        keypoint_brightnesses = dict()
        for kp_idx, kp in keypoints_indexed:
            keypoint_brightnesses[kp_idx] = float(get_star_brightness(kp, workscale_img))

        print("Received {} keypoints.".format(keypoints.__len__()))

        keypoint_indices_remaining = [i for i in range(keypoints.__len__())]

        if "Reduce keypoint amount by splitting the image in several regions near the image borders":
            # Create polygons in 4×3 image regions, than merge the found polygons to remove duplicate polygons.

            img_w = workscale_img.shape[1]
            img_h = workscale_img.shape[0]

            # Dimensions of the image regions at the image's border.
            # This is done to avoid large nonsense polygons spanning over the whole image that will never match
            # a polygon on a corresponding image.
            # If the taken panorama images have a smaller or higher vertical / horizontal overlap,
            # these values should be adjusted accordingly.

            vert_rectangles_h = 0.70  # percent of img_h
            vert_rectangles_w = 0.33  # percent of img_w

            horiztontal_rect_h = 0.33  # percent of img_h
            horiztontal_rect_w = 0.50  # percent of img_w

            if "Debug" == 1:
                print(keypoint_indices_remaining)
                for itm in (206, 237, 292, 204, 201):
                    print(itm in keypoint_indices_remaining)
                    print(keypoints[itm].pt[0])
                    print(keypoints[itm].pt[1])

                print(img_w, img_h)

            keypoint_indices_grouped = {
                # vertical rectangles
                "left_top": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[0] < vert_rectangles_w * img_w and keypoints[kp_idx].pt[
                        1] < vert_rectangles_h * img_h, keypoint_indices_remaining)),
                "left_bot": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[0] < vert_rectangles_w * img_w and keypoints[kp_idx].pt[1] > (
                            1 - vert_rectangles_h) * img_h, keypoint_indices_remaining)),
                "left_mid": list(filter(lambda kp_idx: keypoints[kp_idx].pt[0] < vert_rectangles_w * img_w and abs(
                    keypoints[kp_idx].pt[1] - 0.5 * img_h) < vert_rectangles_h * img_h / 2,
                                        keypoint_indices_remaining)),

                "right_top": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[0] > (1 - vert_rectangles_w) * img_w and keypoints[kp_idx].pt[
                        1] < vert_rectangles_h * img_h, keypoint_indices_remaining)),

                "right_bot": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[0] > (1 - vert_rectangles_w) * img_w and keypoints[kp_idx].pt[
                        1] > (1 - vert_rectangles_h) * img_h, keypoint_indices_remaining)),
                "right_mid": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[0] > (1 - vert_rectangles_w) * img_w and abs(
                        keypoints[kp_idx].pt[1] - 0.5 * img_h) < vert_rectangles_h * img_h / 2,
                    keypoint_indices_remaining)),

                # horizontal rectangles
                "top_left": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[1] < horiztontal_rect_h * img_h and keypoints[kp_idx].pt[
                        0] < horiztontal_rect_w * img_w, keypoint_indices_remaining)),
                "top_right": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[1] < horiztontal_rect_h * img_h and keypoints[kp_idx].pt[0] > (
                            1 - horiztontal_rect_w) * img_w, keypoint_indices_remaining)),
                "top_mid": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[1] < horiztontal_rect_h * img_h and abs(
                        keypoints[kp_idx].pt[0] - 0.5 * img_w) < horiztontal_rect_w * img_w / 2,
                    keypoint_indices_remaining)),

                "bot_left": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[1] > (1 - horiztontal_rect_h) * img_h and keypoints[kp_idx].pt[
                        0] < horiztontal_rect_w * img_w, keypoint_indices_remaining)),
                "bot_right": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[1] > (1 - horiztontal_rect_h) * img_h and keypoints[kp_idx].pt[
                        0] > (1 - horiztontal_rect_w) * img_w, keypoint_indices_remaining)),
                "bot_mid": list(filter(
                    lambda kp_idx: keypoints[kp_idx].pt[1] > (1 - horiztontal_rect_h) * img_h and abs(
                        keypoints[kp_idx].pt[0] - 0.5 * img_w) < horiztontal_rect_w * img_w / 2,
                    keypoint_indices_remaining)),

            }

            # Reduce keypoint amount according to a maximum number of polygons per image
            all_possible_polygons = []
            keep_n_brightest_stars_per_img_region = 15  # start value

            while all_possible_polygons.__len__() == 0 or all_possible_polygons.__len__() > maximum_number_of_polygons_per_image_region:
                print(
                    f"Reducing amount of polygons with: Max. {keep_n_brightest_stars_per_img_region} stars per image region.")

                # Shrink star amount via clustering
                for image_region in keypoint_indices_grouped.keys():
                    indexed_kps_in_this_region = {id: keypoints[id] for id in keypoint_indices_grouped[image_region]}

                    if keep_n_brightest_stars_per_img_region >= list(indexed_kps_in_this_region.keys()).__len__():
                        continue
                    else:
                        # Reduce the maximum star distance step-wise until the number of stars
                        # in this region falls below the maximum allowed number of stars per region.

                        print("Reducing star amount in '{}' from {} to {}.".format(
                            image_region,
                            list(indexed_kps_in_this_region.keys()).__len__(),
                            keep_n_brightest_stars_per_img_region
                        ))

                        remaining_indexed_kps_in_this_region = self.reduce_amount_of_keypoints_by_clustering(
                            indexed_kps_in_this_region,
                            f"{img_filename}__{image_region}.jpg",
                            workscale_img,
                            n_clusters=keep_n_brightest_stars_per_img_region
                        )

                        # Persist values:
                        keypoint_indices_grouped[image_region] = list(remaining_indexed_kps_in_this_region.keys())

                all_possible_polygons = [
                    list(itertools.combinations(keypoint_indices_grouped[grp_key], n_polygon_sides)) for grp_key in
                    keypoint_indices_grouped.keys()
                ]

                # Flatten list and remove duplicates
                all_possible_polygons = set([itm for sublist in all_possible_polygons for itm in sublist])

                print(f"{img_filename}: This yields {all_possible_polygons.__len__()} polygons.")
                print(f"\tKeep the {keep_n_brightest_stars_per_img_region} brightest stars per region.")

                keep_n_brightest_stars_per_img_region -= 1

            print("\n{} are less than the maximum polygon amount of {}, reduction step completed for {}.\n".format(
                format_int(all_possible_polygons.__len__()),
                format_int(maximum_number_of_polygons_per_image_region),
                img_filename,
            ))

            if "Debug" == 1:
                for plg in all_possible_polygons:
                    if all([(kp_id) in plg for kp_id in (206, 237, 292, 204)]):
                        pass

                for plg in all_possible_polygons:
                    if all([(kp_id) in plg for kp_id in (206, 237, 292, 204, 201)]):
                        pass

                for plg in all_possible_polygons:
                    if all([(kp_id) in plg for kp_id in (206, 237, 292, 204, 202)]):
                        pass

        return all_possible_polygons

    def get_polygons_characteristics(self, keypoints, workscale_image, img_title, n_sides, all_possible_polygons):
        """
        Determine the characteristics (angles, side lengths, vertex star brightnesses) of all polygons.

        Parameters
        ----------
        keypoints : tuple(cv2.KeyPoint, cv2.KeyPoint, ...)
            Complete, unreduced set of keypoints from star detection.
            Keypoints from ORB detection are NOT included.

        all_possible_polygons: set([ tuple( keypoint indices ) , ...])
            All possible polygons after reducing the amount of keypoints.


        Returns
        -------
        polygons_characteristics_as_np_array: {
            'keypoint_indices_rotated_list':        {ndarray: (5188,5)},
            'keypoint_x_coordinates_list':          {ndarray: (5188,5)},
            'keypoint_y_coordinates_list':          {ndarray: (5188,5)},
            'keypoint_angles_rotated_list':         {ndarray: (5188,5)},
            'area_list':                            {ndarray: (5188,1)},
            'keypoint_brightnesses_rotated_list':   {ndarray: (5188,5)},
            'side_lengths_rotated_list':            {ndarray: (5188,5)},
        }
            Characteristics of each polygon in a numpy array.
            Differences between polygons will be calculated using numpy operations rather than for loops
            due to performance.
        """

        keypoint_brightnesses = dict()
        for kp_idx, kp in enumerate(keypoints):
            keypoint_brightnesses[kp_idx] = float(get_star_brightness(kp, workscale_image))

        polygons_characteristics_dict = dict()
        for ict, keypoint_indices_unordered in enumerate(all_possible_polygons):

            if ict % 10000 == 0:
                print("Measuring characteristics of {}-keypoint polygon {} of {}".format(
                    n_sides,
                    "{:,}".format(ict).replace(",", "."),
                    "{:,}".format(all_possible_polygons.__len__()).replace(",", "."),

                ))

            if False:
                # Debug
                if all([(kp_id) in keypoint_indices_unordered for kp_id in
                        (206, 237, 292, 204, 201)]):
                    pass

                if all([(kp_id) in keypoint_indices_unordered for kp_id in
                        (206, 237, 292, 204, 202)]):
                    pass

            n_polygon_spanning_points = [keypoints[i].pt for i in keypoint_indices_unordered]

            if False:
                # Debug
                kps_of_interest = [(739.2000122070312, 174.0), (722.0, 247.0), (426.0, 263.0),
                                   (510.0000305175781, 104.4000015258789), (663.8400268554688, 28.80000114440918)]

                if all([abs(itm[0] - itm[1]) < 3 for itm in
                        zip([itm for sl in sorted(kps_of_interest, key=lambda x: x[0]) for itm in sl],
                            [itm for sl in sorted(n_polygon_spanning_points, key=lambda x: x[0]) for itm in sl])]):
                    pass

            # Omit polygons with too short side lengths.
            # TODO: Instead of using pixel distances in the image plane here, angular distances between star-pinhole
            # TODO: vectors should be used.
            if any([self.dist_between_2_points(*pts) < 5 for pts in
                    itertools.combinations(n_polygon_spanning_points, 2)]):
                continue

            # Generate an alpha shape (alpha = 0) (Convex Hull)
            alpha_shape = alphashape.alphashape(n_polygon_spanning_points, 0)  # hull
            if not alpha_shape.boundary:
                continue

            if list(zip(*alpha_shape.exterior.xy)).__len__() < n_sides + 1:
                # There are keypoint(s) that lie inside the polygon
                continue
            else:
                if False:
                    # Debug
                    fig, ax = plt.subplots()
                    ax.scatter(*zip(*n_polygon_spanning_points))
                    # PolygonPatch in descartes:1.1.0 is dysfunctional, therefore it was overridden
                    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
                    plt.show()

            # alphashape's .is_ccw property is dysfunctional. It always returns False.
            if shapely.algorithms.cga.signed_area(alpha_shape.exterior) < 0:
                # counterclockwise
                hull_curve_ccw = list(zip(*alpha_shape.exterior.xy))
            else:
                # clockwise
                # Inverse order
                hull_curve_ccw = list(zip(*alpha_shape.exterior.xy))[::-1]  # In Ordnung

            hull_curve_open = hull_curve_ccw[:-1]  # duplicate start/end point removed

            all_hull_curve_pts_and_their_neighbors = [
                ([hull_curve_ccw[-2]] + hull_curve_ccw)[i:i + 3] for i in range(hull_curve_ccw.__len__() - 1)
            ]

            # Angle of the spherical triangle spanned by 3 stars on an sphere of arbitrary radius in the physical world
            angles_at_points_ccw_order = [
                self.sphere_angle(*pts) for pts in
                all_hull_curve_pts_and_their_neighbors
            ]

            # Put keypoint indices in correct order according to hull curve open
            kp_dict = {kp_idx: keypoints[kp_idx].pt for kp_idx in keypoint_indices_unordered}
            keypoint_indices_according_to_hull_curve_open = []
            for pt in hull_curve_open:
                for kp_idx, keypoint_coordinates in kp_dict.items():
                    if self.dist_between_2_points(pt, keypoint_coordinates) < 0.001:
                        keypoint_indices_according_to_hull_curve_open.append(kp_idx)
            keypoint_indices_according_to_hull_curve_open = tuple(keypoint_indices_according_to_hull_curve_open)
            if keypoint_indices_according_to_hull_curve_open.__len__() != n_sides:
                raise ValueError()

            # Polygon side lengths
            neighboring_points_ccw = [hull_curve_ccw[i:i + 2] for i in range(hull_curve_ccw.__len__() - 1)]
            # Angle distances between stars in the physical world
            side_lengths_adjacent_to_point_in_ccw_direction = list(map(
                self.angle_distance,
                [two_points[0] for two_points in neighboring_points_ccw],
                [two_points[1] for two_points in neighboring_points_ccw]
            )
            )

            # Rotate results so that point with smallest angle is 0th element in list
            n_rotate = angles_at_points_ccw_order.index(min(angles_at_points_ccw_order))

            points_ccw_order_rotated = hull_curve_open[n_rotate:] + hull_curve_open[:n_rotate]
            angles_at_points_ccw_order_rotated = angles_at_points_ccw_order[n_rotate:] + angles_at_points_ccw_order[
                                                                                         :n_rotate]
            keypoint_indices_rotated = keypoint_indices_according_to_hull_curve_open[
                                       n_rotate:] + keypoint_indices_according_to_hull_curve_open[:n_rotate]

            keypoint_brightnesses_rotated = [keypoint_brightnesses[kp_idx] for kp_idx in
                                             keypoint_indices_rotated]

            side_lengths_rotated = side_lengths_adjacent_to_point_in_ccw_direction[
                                   n_rotate:] + side_lengths_adjacent_to_point_in_ccw_direction[:n_rotate]

            polygons_characteristics_dict[keypoint_indices_according_to_hull_curve_open] = {
                "keypoint_indices_rotated": keypoint_indices_rotated,
                # "area": alpha_shape.area,
                "points_ccw_order_rotated": points_ccw_order_rotated,
                "angles_at_points_ccw_order_rotated": angles_at_points_ccw_order_rotated,
                "keypoint_brightnesses_rotated": keypoint_brightnesses_rotated,
                "side_lengths_rotated": side_lengths_rotated,
            }

        print("{} polygons remain.\n".format(polygons_characteristics_dict.keys().__len__()))

        # Prepare for builing np array
        polygons_characteristics_as_lists = {
            "keypoint_indices_rotated_list": [],
            "keypoint_x_coordinates_list": [],
            "keypoint_y_coordinates_list": [],
            "keypoint_angles_rotated_list": [],
            # "area_list": [],
            "keypoint_brightnesses_rotated_list": [],
            "side_lengths_rotated_list": [],
        }

        for kp_indices, data_dict in polygons_characteristics_dict.items():
            polygons_characteristics_as_lists["keypoint_indices_rotated_list"].append(
                list(data_dict["keypoint_indices_rotated"]))
            polygons_characteristics_as_lists["keypoint_x_coordinates_list"].append(
                [pt[0] for pt in data_dict["points_ccw_order_rotated"]])
            polygons_characteristics_as_lists["keypoint_y_coordinates_list"].append(
                [pt[1] for pt in data_dict["points_ccw_order_rotated"]])
            polygons_characteristics_as_lists["keypoint_angles_rotated_list"].append(
                data_dict["angles_at_points_ccw_order_rotated"])
            # polygons_characteristics_as_lists["area_list"].append([data_dict["area"]])
            polygons_characteristics_as_lists["keypoint_brightnesses_rotated_list"].append(
                data_dict["keypoint_brightnesses_rotated"])
            polygons_characteristics_as_lists["side_lengths_rotated_list"].append(data_dict["side_lengths_rotated"])

        # Convert to numpy arrays
        polygons_characteristics_as_np_array = dict()
        for k in polygons_characteristics_as_lists.keys():
            polygons_characteristics_as_np_array[k] = np.array(polygons_characteristics_as_lists[k])

        return polygons_characteristics_as_np_array

    def match(self,
              ft1_kp, ft2_kp,
              ft1_kp_orb_len, ft2_kp_orb_len,
              workscale_img_1, workscale_img_2,
              polygon_data_store,
              img_id_1, img_id_2,
              fts_calculated_counter
              ):
        """
        Matches two sets of features by recognizing identical polygons in both starry sky images.
        The matches returned by this method will then be used to calculate the homography matrix between the
        two pictures.

        Counterpart / alternative to cv2.BFMatcher.match

        Parameters
        ----------
        ft1_kp: tuple(cv2.KeyPoint, cv2.KeyPoint, ...)
            ORB detected keypoints of length 'ft1_kp_orb_len' PLUS the star detected keypoints

        ft2_kp: tuple(cv2.KeyPoint, cv2.KeyPoint, ...)
            ORB detected keypoints of length 'ft1_kp_orb_len' PLUS the star detected keypoints

        polygon_data_store: dict()
            As an attribute of the CustomBruteForceMatcher instance, that calls this function,
            this dict stores all found 'all_possible_polygons_ftx' and 'polygons characteristics'
            so they don't have to be calculated more than once.

        img_id_1: int
            Will be used as the dict key in polygon_data_store.

        img_id_2: int
            Will be used as the dict key in polygon_data_store.


        Returns
        -------
        DMatches: tuple(
            cv2.DMatch,
            cv2.DMatch,
            {
                "distance": 72.0,
                imgIdx: 0,
                queryIdx: 2, # Related src keypoint id in ft1_kp
                trainIdx: 87 # Related dst keypoint id in ft2_kp
            },
            ...
            )

            Valid matches between keypoints.

        """

        keypoints_example = (
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

        # Sides of the polygon
        n_sides = 5  # Not tested with other values. Triangles (n_sides=3) do not work since they are too unspecific.

        if "Feature 1":
            if img_id_1 not in polygon_data_store.keys():
                fts_calculated_counter.append(img_id_1)
                # Skip ORB detected keypoints, work with stars only
                ft1_kp_orb_removed = ft1_kp[ft1_kp_orb_len:]
                all_possible_polygons_ft1 = self.reduce_amount_of_keypoints_and_create_polygon_spanning_tuples_of_keypoints(
                    ft1_kp_orb_removed,
                    self.filename_img1,
                    workscale_img_1,
                    n_sides
                )

                # Get polygon characteristics
                polygons_from_ft1 = self.get_polygons_characteristics(ft1_kp_orb_removed, workscale_img_1,
                                                                      "img_title_1", n_sides, all_possible_polygons_ft1)

                # Normalize angles
                polygons_from_ft1["keypoint_angles_rotated_list"] /= math.pi

                # Normalize side lengths
                # (Maximum side length of *each* polygon will be 1!)
                polygons_from_ft1["side_lengths_rotated_list"] = (
                        polygons_from_ft1["side_lengths_rotated_list"].T / np.amax(
                    polygons_from_ft1["side_lengths_rotated_list"], axis=1)).T

                if False:
                    # Areas are ignored.
                    # Do NOT normalize areas!
                    polygons_from_ft1["area_list"] = np.sqrt(polygons_from_ft1["area_list"])  # Un-square area

                # Normalize star brightnesses
                # (Maximum star brightnesses within *each* polygon will be 1!)
                polygons_from_ft1["keypoint_brightnesses_rotated_list"] = (
                        polygons_from_ft1["keypoint_brightnesses_rotated_list"].T / np.amax(
                    polygons_from_ft1["keypoint_brightnesses_rotated_list"], axis=1)).T

                # Store data so it doesn't have to be calculated again for other image combinations,
                # in which this image appears
                polygon_data_store[img_id_1] = {
                    "ftX_kp_orb_removed": ft1_kp_orb_removed,
                    "all_possible_polygons_ftX": all_possible_polygons_ft1,
                    "polygons_from_ftX": polygons_from_ft1,
                }
            else:
                # Data was already calculated before.
                ft1_kp_orb_removed = polygon_data_store[img_id_1]["ftX_kp_orb_removed"]
                all_possible_polygons_ft1 = polygon_data_store[img_id_1]["all_possible_polygons_ftX"]
                polygons_from_ft1 = polygon_data_store[img_id_1]["polygons_from_ftX"]

        if "Feature 2":
            if img_id_2 not in polygon_data_store.keys():
                fts_calculated_counter.append(img_id_2)
                # Skip ORB detected keypoints, work with stars only
                ft2_kp_orb_removed = ft2_kp[ft2_kp_orb_len:]
                all_possible_polygons_ft2 = self.reduce_amount_of_keypoints_and_create_polygon_spanning_tuples_of_keypoints(
                    ft2_kp_orb_removed,
                    self.filename_img2,
                    workscale_img_2,
                    n_sides
                )

                # Get polygon characteristics
                polygons_from_ft2 = self.get_polygons_characteristics(ft2_kp_orb_removed, workscale_img_2,
                                                                      "img_title_2", n_sides, all_possible_polygons_ft2)

                # Normalize angles
                polygons_from_ft2["keypoint_angles_rotated_list"] /= math.pi

                # Normalize side lengths
                # (Maximum side length of *each* polygon will be 1!)
                polygons_from_ft2["side_lengths_rotated_list"] = (
                        polygons_from_ft2["side_lengths_rotated_list"].T / np.amax(
                    polygons_from_ft2["side_lengths_rotated_list"], axis=1)).T

                if False:
                    # Areas are ignored.
                    # Do NOT normalize areas!
                    polygons_from_ft2["area_list"] = np.sqrt(polygons_from_ft2["area_list"])  # Un-square area

                # Normalize star brightnesses
                # (Maximum star brightnesses within *each* polygon will be 1!)
                polygons_from_ft2["keypoint_brightnesses_rotated_list"] = (
                        polygons_from_ft2["keypoint_brightnesses_rotated_list"].T / np.amax(
                    polygons_from_ft2["keypoint_brightnesses_rotated_list"], axis=1)).T

                # Store data so it doesn't have to be calculated again for other image combinations,
                # in which this image appears
                polygon_data_store[img_id_2] = {
                    "ftX_kp_orb_removed": ft2_kp_orb_removed,
                    "all_possible_polygons_ftX": all_possible_polygons_ft2,
                    "polygons_from_ftX": polygons_from_ft2,
                }
            else:
                # Data was already calculated before.
                ft2_kp_orb_removed = polygon_data_store[img_id_2]["ftX_kp_orb_removed"]
                all_possible_polygons_ft2 = polygon_data_store[img_id_2]["all_possible_polygons_ftX"]
                polygons_from_ft2 = polygon_data_store[img_id_2]["polygons_from_ftX"]

        if "Plot *all* keypoints (before clustering)":
            # Add the length of orb-detected keypoints from classic orb-matcher, which will
            # preceed the star keypoints in the final ImageFeature2D object

            ft1_kp_orb_removed_indexed_increased = [(kp_idx + ft1_kp_orb_len, kp) for kp_idx, kp in
                                                    enumerate(ft1_kp_orb_removed)]
            ft2_kp_orb_removed_indexed_increased = [(kp_idx + ft2_kp_orb_len, kp) for kp_idx, kp in
                                                    enumerate(ft2_kp_orb_removed)]

            self.imwrite_keypoints_of_2_images_to_disk(
                ft1_kp_orb_removed_indexed_increased,
                ft2_kp_orb_removed_indexed_increased,
                workscale_img_1,
                workscale_img_2,
                "00_all_star_keypoints_before_clustering_{}___{}__{}kps___{}kps.jpg".format(
                    self.filename_img1,
                    self.filename_img2,
                    ft1_kp_orb_removed_indexed_increased.__len__(),
                    ft2_kp_orb_removed_indexed_increased.__len__(),
                ),
                circle_color=(0, 0, 255),
                text_color=(0, 0, 150),
                scale_circle=3
            )

        if "Plot remaining keypoints (after clustering)":
            # Flatten:
            kp_indices_remaining_ft1 = set(
                (kp_idx for polygon_spanning_tpl in all_possible_polygons_ft1 for kp_idx in polygon_spanning_tpl))
            kp_indices_remaining_ft2 = set(
                (kp_idx for polygon_spanning_tpl in all_possible_polygons_ft2 for kp_idx in polygon_spanning_tpl))

            # Add the length of orb-detected keypoints from classic orb-matcher, which will
            # preceed the star keypoints in the final ImageFeature2D object

            kps_ft1_remaining_indexed_increased = [(kp_idx + ft1_kp_orb_len, ft1_kp_orb_removed[kp_idx]) for kp_idx in
                                                   kp_indices_remaining_ft1]
            kps_ft2_remaining_indexed_increased = [(kp_idx + ft2_kp_orb_len, ft2_kp_orb_removed[kp_idx]) for kp_idx in
                                                   kp_indices_remaining_ft2]

            self.imwrite_keypoints_of_2_images_to_disk(
                kps_ft1_remaining_indexed_increased,
                kps_ft2_remaining_indexed_increased,
                workscale_img_1,
                workscale_img_2,
                "02_remaining_keypoints_after_clustering__{}___{}__{}kps___{}kps.jpg".format(
                    self.filename_img1,
                    self.filename_img2,
                    kps_ft1_remaining_indexed_increased.__len__(),
                    kps_ft2_remaining_indexed_increased.__len__(),
                ),
                circle_color=(0, 0, 255),
                text_color=(0, 0, 150),
                scale_circle=3
            )

        if "Plot *all* keypoints (before clustering) and highlight keypoints removed by clustering":
            # Add the length of orb-detected keypoints from classic orb-matcher, which will
            # preceed the star keypoints in the final ImageFeature2D object

            labels = {
                "removed": 0,  # Keypoint removed by clustering
                "remaining": 1,  # Keypoint remaining after clustering
            }
            ft1_kp_orb_removed_indexed_increased = [[kp_idx + ft1_kp_orb_len, kp, labels["removed"]] for kp_idx, kp in
                                                    enumerate(ft1_kp_orb_removed)]
            ft2_kp_orb_removed_indexed_increased = [[kp_idx + ft2_kp_orb_len, kp, labels["removed"]] for kp_idx, kp in
                                                    enumerate(ft2_kp_orb_removed)]

            for kp_info_list in ft1_kp_orb_removed_indexed_increased:
                if kp_info_list[0] in [kp_idx + ft1_kp_orb_len for kp_idx in kp_indices_remaining_ft1]:
                    kp_info_list[2] = labels["remaining"]

            for kp_info_list in ft2_kp_orb_removed_indexed_increased:
                if kp_info_list[0] in [kp_idx + ft2_kp_orb_len for kp_idx in kp_indices_remaining_ft2]:
                    kp_info_list[2] = labels["remaining"]

            self.imwrite_keypoints_of_2_images_to_disk(
                ft1_kp_orb_removed_indexed_increased,
                ft2_kp_orb_removed_indexed_increased,
                workscale_img_1,
                workscale_img_2,
                "01_all_star_keypoints_highlighted_if_removed_by_clustering_{}___{}__{}kps___{}kps.jpg".format(
                    self.filename_img1,
                    self.filename_img2,
                    ft1_kp_orb_removed_indexed_increased.__len__(),
                    ft2_kp_orb_removed_indexed_increased.__len__(),
                ),
                circle_color=(0, 0, 255),
                text_color=(0, 0, 150),
                scale_circle=3
            )

        print("Matching polygons")
        if False:
            # Init similartiy matrix
            polygon_similarity_matrix = np.zeros(
                (
                    polygons_from_ft1["keypoint_indices_rotated_list"].shape[0],
                    polygons_from_ft2["keypoint_indices_rotated_list"].shape[0],
                )
            )

            num_polygon_combinations = polygons_from_ft1["keypoint_indices_rotated_list"].shape[0] * \
                                       polygons_from_ft2["keypoint_indices_rotated_list"].shape[0]

        if False:
            # Calculate distances between polygons
            # Time-consuming for loops
            for r in range(polygons_from_ft1["keypoint_indices_rotated_list"].shape[0]):
                for c in range(polygons_from_ft2["keypoint_indices_rotated_list"].shape[0]):
                    if (r * polygons_from_ft2["keypoint_indices_rotated_list"].shape[0] + c) % 100000 == 0:
                        print("Calculating similiraties for polygon combination {} of {}".format(
                            "{:,}".format(r * polygons_from_ft2["keypoint_indices_rotated_list"].shape[0] + c).replace(
                                ",", "."),
                            "{:,}".format(num_polygon_combinations).replace(",", "."),
                        ))
                    similartiy = math.sqrt(
                        np.sum(np.square(polygons_from_ft1["keypoint_angles_rotated_list"][r, :] - polygons_from_ft2[
                                                                                                       "keypoint_angles_rotated_list"][
                                                                                                   c, :]))
                        + np.sum(np.square(polygons_from_ft1["side_lengths_rotated_list"][r, :] - polygons_from_ft2[
                                                                                                      "side_lengths_rotated_list"][
                                                                                                  c, :]))
                        + np.sum(np.square(polygons_from_ft1["area_list"][r, :] - polygons_from_ft2["area_list"][c, :]))
                        + np.sum(np.square(
                            polygons_from_ft1["keypoint_brightnesses_rotated_list"][r, :] - polygons_from_ft2[
                                                                                                "keypoint_brightnesses_rotated_list"][
                                                                                            c, :]))

                    )
                    polygon_similarity_matrix[r, c] = similartiy
        if False:
            # Calculate distances between polygons
            # for loop + np.array (faster but still slow)
            for c in range(polygons_from_ft2["keypoint_indices_rotated_list"].shape[0]):
                if c % 100 == 0:
                    print("Calc similarity matrix col {} of {}".format(c, polygons_from_ft2[
                        "keypoint_indices_rotated_list"].shape[0]))
                idx_of_poly_from_ft2 = c
                dist_angles = np.sum(np.square(polygons_from_ft1["keypoint_angles_rotated_list"] - polygons_from_ft2[
                                                                                                       "keypoint_angles_rotated_list"][
                                                                                                   idx_of_poly_from_ft2,
                                                                                                   :]), axis=1)
                dist_lengths = np.sum(np.square(
                    polygons_from_ft1["side_lengths_rotated_list"] - polygons_from_ft2["side_lengths_rotated_list"][
                                                                     idx_of_poly_from_ft2, :]), axis=1)
                # dist_areas = np.sum(np.square(polygons_from_ft1["area_list"]-polygons_from_ft2["area_list"][idx_of_poly_from_ft2,:]),axis=1)
                dist_brightnesses = np.sum(np.square(
                    polygons_from_ft1["keypoint_brightnesses_rotated_list"] - polygons_from_ft2[
                                                                                  "keypoint_brightnesses_rotated_list"][
                                                                              idx_of_poly_from_ft2, :]), axis=1)
                overall_dist = np.sqrt(
                    dist_angles +
                    dist_lengths +
                    # dist_areas+
                    dist_brightnesses
                )
                polygon_similarity_matrix[:, idx_of_poly_from_ft2] = overall_dist

        if True:
            """
            Distance between two polygons is the Euclidean distance between all its properties:
            distance = math.sqrt(
              (angle_poly1_vertex1 - angle_poly2_vertex1)**2 +
              (angle_poly1_vertex2 - angle_poly2_vertex2)**2 +
              (angle_poly1_vertex3 - angle_poly2_vertex3)**2 +
              ... +
              (side_length_poly1_vertex1 - side_length_poly2_vertex1)**2 +
              (side_length_poly1_vertex2 - side_length_poly2_vertex2)**2 +
              (side_length_poly1_vertex3 - side_length_poly2_vertex3)**2 +
              ... +
              (star_brightness_poly1_vertex1 - star_brightness_poly2_vertex1)**2 +
              (star_brightness_poly1_vertex2 - star_brightness_poly2_vertex2)**2 +
              (star_brightness_poly1_vertex3 - star_brightness_poly2_vertex3)**2 +
              ...
            )
            """

            distances = dict()
            print("\nCalculating polygon differences.")
            print("Differences np array will have {} × {} = {} rows.".format(
                "{:,}".format(polygons_from_ft2["keypoint_indices_rotated_list"].shape[0]).replace(",", "."),
                "{:,}".format(polygons_from_ft1["keypoint_indices_rotated_list"].shape[0]).replace(",", "."),
                "{:,}".format(polygons_from_ft1["keypoint_indices_rotated_list"].shape[0] *
                              polygons_from_ft2["keypoint_indices_rotated_list"].shape[0]).replace(",", "."),
            ))

            for property in (
                    "keypoint_angles_rotated_list",
                    "side_lengths_rotated_list",
                    # "area_list",
                    "keypoint_brightnesses_rotated_list",
            ):
                print("\t– Calculating differences in {}".format(property))

                distances[property] = np.sum(np.square(
                    np.tile(
                        polygons_from_ft1[property],
                        reps=(polygons_from_ft2["keypoint_indices_rotated_list"].shape[0], 1)
                    ) - np.repeat(
                        polygons_from_ft2[property],
                        repeats=polygons_from_ft1["keypoint_angles_rotated_list"].shape[0], axis=0
                    )
                ), axis=1)

            print("\tCalculating overall distances")
            overall_dist = np.sqrt(
                distances["keypoint_angles_rotated_list"] +
                distances["side_lengths_rotated_list"] +
                # distances["area_list"] #+
                distances["keypoint_brightnesses_rotated_list"]
            )
            # Free memory:
            distances = None

            # Common distances for a good match
            # overall = angle + length + brightness =
            # angle = 0.0055
            # side length = 0.0055
            # brightnesses = 0.0142

            print("Assembling polygon_similarity_matrix")
            polygon_similarity_matrix = np.reshape(
                overall_dist,
                (
                    polygons_from_ft2["keypoint_indices_rotated_list"].shape[0],
                    polygons_from_ft1["keypoint_indices_rotated_list"].shape[0]
                )

            ).T

        print("{} <=> {}   Min similarity: {}, max: {}, mean:{}".format(
            self.filename_img1,
            self.filename_img2,
            polygon_similarity_matrix.min(axis=1).min(),
            polygon_similarity_matrix.min(axis=1).max(),
            polygon_similarity_matrix.min(axis=1).mean(),
        ))

        stat_lines = []

        max_polygon_distance_for_disk_output = 1.2  # Polygons with a better euclidian distance will be dumped to disk
        max_polygon_distance_threshold_for_valid_matches = 0.1566  # Determined empirically!

        if max_polygon_distance_for_disk_output < max_polygon_distance_threshold_for_valid_matches:
            max_polygon_distance_for_disk_output = max_polygon_distance_threshold_for_valid_matches

        if False:
            for r, row_min in enumerate(polygon_similarity_matrix.min(axis=1)):
                if row_min < max_polygon_distance_for_disk_output:
                    # Get column position with minimum value
                    c = int(np.where(np.isclose(polygon_similarity_matrix[r, :], row_min))[0][0])

                    if (r, c) == (569, 143):
                        pass

                    list_of_matching_polygon_indices.append((r, c, row_min))

        row_indices = np.arange(polygon_similarity_matrix.shape[0])  # Eqivalent to polygon_ft1 indices
        col_indices_mimima = np.argmin(polygon_similarity_matrix, axis=1)  # Eqivalent to polygon_ft2 indices
        minima_values = polygon_similarity_matrix.min(axis=1)

        polygon_ft1_idx_polygon_ft2_idx = np.stack((row_indices, col_indices_mimima)).T

        if "Collect polygon matches better than max_polygon_distance_for_disk_output for disk output":
            disk_output_polygon_matches_indices = polygon_ft1_idx_polygon_ft2_idx[
                                                  minima_values < max_polygon_distance_for_disk_output, :]
            disk_output_polygon_matches_differences = np.reshape(minima_values, (-1, 1))[
                                                      minima_values < max_polygon_distance_for_disk_output, :]

            disk_output_polygon_matches_and_differences_as_list = tuple(
                map(list.__add__, disk_output_polygon_matches_indices.tolist(),
                    disk_output_polygon_matches_differences.tolist()))

            limit_best_matches = 100
            print("Will write the best {} of {} polygon matches with a distance better than {:.4f} to disk.".format(
                limit_best_matches,
                disk_output_polygon_matches_indices.shape[0],
                max_polygon_distance_for_disk_output
            )
            )

            for i, polygon_ft1_ft2_tpl in enumerate(
                    sorted(disk_output_polygon_matches_and_differences_as_list, key=lambda tpl: tpl[2])[
                    :limit_best_matches]):
                keypoint_indices_spanning_polygon_ft1 = tuple(
                    polygons_from_ft1['keypoint_indices_rotated_list'][polygon_ft1_ft2_tpl[0], :])
                keypoint_indices_spanning_polygon_ft2 = tuple(
                    polygons_from_ft2['keypoint_indices_rotated_list'][polygon_ft1_ft2_tpl[1], :])

                if False:
                    # Debug
                    if any([
                        kp_idx not in keypoint_indices_spanning_polygon_ft1 for kp_idx in (237, 211, 202, 248, 232)
                    ]):
                        continue

                if (i + 1) % 10 == 0:
                    print("Dumping polygon match {} of best {} of {} to disk.".format(
                        i + 1,
                        limit_best_matches,
                        disk_output_polygon_matches_and_differences_as_list.__len__()
                    ))

                # polygon_ft1_ft2_tpl[0]: row idx in polygons_from_ft1
                # polygon_ft1_ft2_tpl[1]: row idx in polygons_from_ft2
                # polygon_ft1_ft2_tpl[2]: Distance between polygons

                if True:
                    self.imwrite_keypoints_of_2_images_to_disk(
                        tuple((idx + ft1_kp_orb_len, ft1_kp_orb_removed[idx]) for idx in
                              keypoint_indices_spanning_polygon_ft1),
                        tuple((idx + ft2_kp_orb_len, ft2_kp_orb_removed[idx]) for idx in
                              keypoint_indices_spanning_polygon_ft2),

                        workscale_img_1,
                        workscale_img_2,
                        "dist={:.4f}__polygon-{}_and_polygon{}__spanned_by_kps({})_and({}).jpg".format(
                            polygon_ft1_ft2_tpl[2],  # distance

                            polygon_ft1_ft2_tpl[0],  # poly1 id
                            polygon_ft1_ft2_tpl[1],  # poly2 id

                            "-".join(
                                [str(itm + ft1_kp_orb_len) for itm in list(keypoint_indices_spanning_polygon_ft1)]),
                            "-".join(
                                [str(itm + ft2_kp_orb_len) for itm in list(keypoint_indices_spanning_polygon_ft2)]),

                        ),
                        circle_color=(154, 244, 0)[::-1],
                        text_color=(47, 74, 0)[::-1],
                        star_brightnesses_normalized_poly1=polygons_from_ft1['keypoint_brightnesses_rotated_list'][
                            polygon_ft1_ft2_tpl[0]],
                        star_brightnesses_normalized_poly2=polygons_from_ft2['keypoint_brightnesses_rotated_list'][
                            polygon_ft1_ft2_tpl[1]],
                        distance_between_displayed_2_polygons=polygon_ft1_ft2_tpl[2],  # distance,
                        max_polygon_distance_threshold_for_valid_matches=max_polygon_distance_threshold_for_valid_matches,
                    )

        if "Collect valid polygon matches (better than max_polygon_distance_threshold_for_valid_matches) ":
            valid_polygon_matches_indices = polygon_ft1_idx_polygon_ft2_idx[
                                            minima_values < max_polygon_distance_threshold_for_valid_matches, :]
            valid_polygon_matches_differences = np.reshape(minima_values, (-1, 1))[
                                                minima_values < max_polygon_distance_threshold_for_valid_matches, :]

            valid_polygon_matches_and_differences_as_list = tuple(
                map(list.__add__, valid_polygon_matches_indices.tolist(),
                    valid_polygon_matches_differences.tolist()))

            print("Remaining {} polygon matches with a distance better than {:.4f}.".format(
                valid_polygon_matches_indices.shape[0],
                max_polygon_distance_threshold_for_valid_matches
            )
            )

        if "Assemble result object (list of DMatches)":
            matching_keypoints_ft1_and_ft2 = defaultdict(list)

            for polygon_ft1_ft2_tpl in valid_polygon_matches_and_differences_as_list:

                # polygon_tpl[0]: row idx in polygons_from_ft1
                # polygon_tpl[1]: row idx in polygons_from_ft2
                # polygon_tpl[2]: Distance between polygons

                for kp_poly_ft1_kp_poly_ft2_tuple in zip(
                        # Keypoints of this polygon from ft1
                        list(polygons_from_ft1['keypoint_indices_rotated_list'][polygon_ft1_ft2_tpl[0]]),
                        # match keypoints of this poloygon from ft2
                        list(polygons_from_ft2['keypoint_indices_rotated_list'][polygon_ft1_ft2_tpl[1]])
                ):
                    matching_keypoints_ft1_and_ft2[kp_poly_ft1_kp_poly_ft2_tuple].append(polygon_ft1_ft2_tpl[2])

            matching_keypoints_ft1_and_ft2__example = {
                # (kp_ft1_idx, kp_ft2_idx): [ differences per match ]
                (28, 53): [0.15095320241450447, 0.14045850422107323],
                (4, 4): [0.15095320241450447, 0.03328439129379918],
                (58, 74): [0.15095320241450447, 0.14045850422107323, 0.15622986276105214],
                (10, 6): [0.15095320241450447], (46, 51): [0.15095320241450447],
                (34, 48): [0.14045850422107323],
            }

            # Order by number of matches per keypoint combinations.
            matching_keypoints_ordered = sorted(
                [(kp_ids, distances.__len__(), distances) for kp_ids, distances in
                 matching_keypoints_ft1_and_ft2.items()],
                key=lambda x: x[1]
            )[::-1]

            demo_result_of_match_method = (
                "< cv2.DMatch 0x7f15fc951890>",
                "< cv2.DMatch 0x7f15fc953f90>",
                {
                    "distance": 61.0,
                    "imgIdx": 0,
                    "queryIdx": 37,  # max: 5513 => Index des Keypoints in ft1
                    "trainIdx": 211,  # max: 4435 ==> Index des Keypoints in ft2
                }
            )

            DMatches = []

            # The amount of polygons in which the keypoint has to match in order to be considered as a valid keypoint match.
            two_keypoints_have_to_match_in_at_least_n_polygons = 1

            for kp_poly_ft1_kp_poly_ft2_tuple, num_matches, distances in filter(
                    lambda x: x[1] >= two_keypoints_have_to_match_in_at_least_n_polygons, matching_keypoints_ordered):
                new_DMatch = cv.DMatch()
                new_DMatch.distance = sum(distances) / num_matches  # TODO: Does this value have an impact later?
                new_DMatch.imgIdx = 0  # train image index (Value is irrelevant!)
                new_DMatch.queryIdx = kp_poly_ft1_kp_poly_ft2_tuple[0] + ft1_kp_orb_len  # query descriptor index
                new_DMatch.trainIdx = kp_poly_ft1_kp_poly_ft2_tuple[1] + ft2_kp_orb_len  # train descriptor index

                DMatches.append(new_DMatch)

            print("DMatches.__len__() = {}".format(DMatches.__len__()))

            return tuple(DMatches)

    def imwrite_keypoints_of_2_images_to_disk(
            self,
            keypoints_ft1_with_orb_increased_idx, keypoints_ft2_with_orb_increased_idx,
            img1, img2,
            filename,
            circle_color,
            text_color,
            scale_circle=1,
            star_brightnesses_normalized_poly1=None,
            star_brightnesses_normalized_poly2=None,
            distance_between_displayed_2_polygons=None,
            max_polygon_distance_threshold_for_valid_matches=None,
    ):
        def contourArea_to_circle_radius(contour_area):
            x1 = 24  # contourArea
            y1 = 15  # target radius

            x2 = 0  # contourArea
            y2 = 5  # target radius

            return max(1, int((y2 - y1) / (x2 - x1) * (contour_area - x1) + y1)) * scale_circle

        circle_color_from_label = {
            # removed keypoint
            0: (251, 155, 55)[::-1],  # orange
            # remaining keypoint
            1: circle_color,
        }

        text_color_from_label = {
            # removed keypoint
            0: (135, 68, 0)[::-1],  # dark orange
            # remaining keypoint
            1: text_color,
        }

        out_image_w_matches = cv.hconcat([img1, img2])

        out_image_w_matches = optimize_img_for_feature_detection(out_image_w_matches)
        out_image_w_matches = cv.cvtColor(out_image_w_matches, cv.COLOR_GRAY2BGR)

        for im_idx, img_name in enumerate([self.filename_img1, self.filename_img2]):
            x_shift = {
                0: 0,
                1: img1.shape[1]
            }[im_idx]

            cv.putText(
                out_image_w_matches,
                text=str(img_name),
                # 	Bottom-left corner of the text string in the image.
                org=(
                    int(20 + x_shift),
                    int(img1.shape[0] - 30)
                ),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                # color=(222,255,17)[::-1], # yellow
                color=(222, 255, 255)[::-1],  # white
                thickness=1  #
            )

        for ft_id, indexed_orb_increased_keypoints in enumerate(
                [keypoints_ft1_with_orb_increased_idx, keypoints_ft2_with_orb_increased_idx]):
            x_shift = {
                0: 0,
                1: img1.shape[1]
            }[ft_id]

            for i, tpl in enumerate(indexed_orb_increased_keypoints):
                if tpl.__len__() == 3:
                    kp_id, kp, label = tpl
                elif tpl.__len__() == 2:
                    kp_id, kp = tpl
                    label = 1  # default color ( remaining keypoint)

                # radius = int((5 / 8) * kp.size + 15 * 5 / 8)
                if star_brightnesses_normalized_poly1 is not None and star_brightnesses_normalized_poly2 is not None:
                    radius = [
                                 star_brightnesses_normalized_poly1,  # [0...1]
                                 star_brightnesses_normalized_poly2,
                             ][ft_id][i] * 30
                else:
                    radius = contourArea_to_circle_radius(kp.size)  # contourArea was stored in kp.size

                out_image_w_matches = cv.circle(
                    out_image_w_matches,
                    (int(kp.pt[0] + x_shift), int(kp.pt[1])),
                    radius=int(radius),
                    # radius=max(1,int(kp.size)), # contourArea was stored in kp.size
                    color=circle_color_from_label[label],
                    thickness=1,
                    lineType=cv.LINE_AA  # cv.FILLED # cv.LINE_4 cv.LINE_8 cv.LINE_AA
                )
                cv.putText(
                    out_image_w_matches,
                    text=str(kp_id),
                    # text=str("{:.4f}".format(kp.size)),
                    # 	Bottom-left corner of the text string in the image.
                    org=(
                        int(kp.pt[0] + x_shift + 5 + radius),
                        int(kp.pt[1] - 5 - radius)
                    ),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=text_color_from_label[label],
                    thickness=1  #
                )

        # Print the distance and distance threshold for the 2 polygons displayed:
        if all([
            distance_between_displayed_2_polygons is not None,
            max_polygon_distance_threshold_for_valid_matches is not None,
        ]):
            cv.putText(
                out_image_w_matches,
                text="[{}] Euclidean distance = {:.6f} {} threshold ({:.6f})".format(
                    "ACCEPTED" if distance_between_displayed_2_polygons < max_polygon_distance_threshold_for_valid_matches else "OMiTTED",
                    distance_between_displayed_2_polygons,
                    "<" if distance_between_displayed_2_polygons < max_polygon_distance_threshold_for_valid_matches else ">",
                    max_polygon_distance_threshold_for_valid_matches
                ),
                # 	Bottom-left corner of the text string in the image.
                org=(
                    int(25),
                    int(img1.shape[0] - 70)
                ),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(154, 244, 0)[
                      ::-1] if distance_between_displayed_2_polygons < max_polygon_distance_threshold_for_valid_matches else (
                                                                                                                                 206,
                                                                                                                                 97,
                                                                                                                                 148)[
                                                                                                                             ::-1],
                thickness=1  #
            )

        cv.imwrite(
            os.path.join(
                self.output_dir,
                filename

            ),
            out_image_w_matches
        )


def format_int(int_value):
    """
    Format using "." as thousands separator:
    12301 => 12.301

    """
    return "{:,}".format(int_value).replace(",", ".")

# Joachim Broser 2022