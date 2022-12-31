# OpenCV Starry Sky Panorama Stitcher

## Motivation

- PTGui (https://ptgui.com/) does an excellent stitching job but it does not recognize stars as features. On each combination of 2 starry sky image, at least 3 features on each have to be labeled **manually**. Taking 21 images of the night sky yields at least 36 overlaps between images. For each overlap at least 3 stars have to be labeled manually on both images of the overlap. Pretty annoying.
  ![](docs/pt_gui_manual_keypoint_labeling.png)
- https://github.com/opencv/opencv/blob/4.x/samples/python/stitching_detailed.py is a great example but not very self-explanatory for someone who hasn't stitched panoramas with OpenCV before. The stitching pipeline presented in `stitching_detaily.py` (comprising feature detection, pairwise image matching, homography estimation, bundle adjustment, waviness correction, image warping, stitching seam estimation and seaming/masking, timelapsing of warped images, blending of warped images) does not provide deep insights into what OpenCV is doing under the hood and what is the result/output of each step. 
  **Remedy is provided here:** The stitching pipeline presented here is an enhanced version of `stitching_detaily.py` and offers more documentation within the code and a better understanding of each step within the stitching pipeline due to a lot of image output to disk.
- The matchers used in  `stitching_detaily.py`  (`cv.detail_BestOf2NearestMatcher` and `cv.detail_BestOf2NearestRangeMatcher`) perform poorly on starry sky images. **Therefore a custom bruteforce matcher** (leveraging `cv.BFMatcher(cv.NORM_HAMMING)`) **is introduced here** which returns a tuple of `cv2.detail.MatchesInfo` objects like `cv.detail_BestOf2NearestMatcher` and `cv.detail_BestOf2NearestRangeMatcher` also do.

## Collecting images for a fisheye panorama

For an all sky fisheye panorama starry sky images should be taken according to a plan.

For an 18 mm lens this setup yields good overlap between the images:

![](docs/image_plan_for_all_sky_circular_fisheye.png)

## ORB keypoint/descriptor matching vs. shape/constellation matching

OpenCV matchers can match keypoints/descriptors, which are single points on each image:

![](docs/Oystercatcher_01_full.png)

![](docs/Oystercatcher_02_keypoints.png)

On starry sky images, especially on images with a low amount of stars and a dark unique background, this can become problematic because »many stars look the same«.

Matching star constellations would be helfpful but OpenCV matchers are not able to match constellations – they are just able to match keypoints/descriptors.

The **StarPolygonMatcher** presented here provides this functionality:

![](docs/polygon_shape_matching.jpg)

Star constellations or polygons are compared based on angles, side lengths and star brightnesses.

For a 5-sided polygon like shown above there are 5 vertex angles, 5 side lengths and 5 star brightnesses.

The **vertex angles and side lengths** are **not measured in the image plane**!  The stars from the image plane are projected to a sphere in the physical real world. Angles and side lengths are then measured on this sphere leveraging spherical trigonometry. The radius of this sphere does not matter since **side lengths of spherical triangles** are measured in **radian**.

![](docs/sphere.png)

## The panorama stitching pipeline

![](docs/pipeline.png)

## Examples

### Example 1: Stitching daylight images

`cv.detail_BestOf2NearestMatcher` (left) and `CustomBruteForceMatcher` (right) yield comparable results on daylight images.

![](docs/example_01/cv.detail_BestOf2NearestMatcher_vs_CustomBruteForceMatcher.jpg)

### Example 2: Colorizing seams and edges

#### Keypoints detected by ORB

![](docs/example_02_colorized_seams_and_edges/z_ORB_keypoints.jpg)

#### Stars detected by Canny Edge

![](docs/example_02_colorized_seams_and_edges/z_CannyEdge.jpg)

#### ORB features generated from stars detected by Canny Edge

![](docs/example_02_colorized_seams_and_edges/z_StarFeatures.jpg)

#### Matches (inliers) found via ORB keypoing matching

![](docs/example_02_colorized_seams_and_edges/ORB___omitted_____conf=0.22444___num_inliers=9___02-horiz-ne.jpg___10-alt1-ne.jpg.jpg)

#### Reducing the amount of stars by `sklearn.cluster.AgglomerativeClustering`

All keypoints created by Canny Edge star detection:

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/00_all_star_keypoints_before_clustering_02-horiz-ne.jpg___10-alt1-ne.jpg__619kps___1000kps.jpg)

Keypoints marked for deletion after a cluster size of 15 stars per image region was defined:

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/01_all_star_keypoints_highlighted_if_removed_by_clustering_02-horiz-ne.jpg___10-alt1-ne.jpg__619kps___1000kps.jpg)

The remaining stars after clustering:

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/02_remaining_keypoints_after_clustering__02-horiz-ne.jpg___10-alt1-ne.jpg__68kps___72kps.jpg)

Star amount was reduced by splitting the image in several regions and reducing the size of clusters per region simultaneously by `sklearn.cluster.AgglomerativeClustering`:

![](docs/cluster_star_reduction_steps.png)

The 12 image regions used for star clustering are:

![](docs/12_image_regions_for_clustering.png)

#### Matching polygons (star constellations) found by the Custom Star Polygon Matcher

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0620__polygon-1708_and_polygon1134__spanned_by_kps(1098-1010-1016-1020-1197)_and(1108-1022-1036-1050-1169).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0653__polygon-2730_and_polygon815__spanned_by_kps(1025-1201-1098-1125-1197)_and(1041-1147-1108-1149-1169).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0733__polygon-466_and_polygon3638__spanned_by_kps(1010-1222-1013-1000-1020)_and(1022-1233-1020-1005-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0736__polygon-765_and_polygon3558__spanned_by_kps(1227-1098-1010-1016-1020)_and(1176-1108-1022-1036-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0758__polygon-2560_and_polygon2061__spanned_by_kps(1227-1010-1016-1020-1197)_and(1176-1022-1036-1050-1169).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0796__polygon-793_and_polygon2716__spanned_by_kps(1013-1000-1020-1016-1222)_and(1020-1005-1050-1036-1233).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0835__polygon-1490_and_polygon46__spanned_by_kps(1138-1010-1222-1000-1020)_and(1093-1022-1233-1005-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0839__polygon-386_and_polygon860__spanned_by_kps(1227-1098-1010-1016-1197)_and(1176-1108-1022-1036-1169).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0847__polygon-2016_and_polygon2421__spanned_by_kps(1227-1098-1010-1020-1197)_and(1176-1108-1022-1050-1169).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0867__polygon-1229_and_polygon3208__spanned_by_kps(1138-1016-1222-1000-1020)_and(1093-1036-1233-1005-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0881__polygon-1506_and_polygon1748__spanned_by_kps(1138-1010-1222-1013-1000)_and(1093-1022-1233-1020-1005).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0881__polygon-2053_and_polygon2543__spanned_by_kps(1138-1222-1013-1000-1020)_and(1093-1233-1020-1005-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0895__polygon-194_and_polygon2142__spanned_by_kps(1138-1016-1222-1013-1000)_and(1093-1036-1233-1020-1005).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0916__polygon-1952_and_polygon1434__spanned_by_kps(1000-1020-1138-1010-1016)_and(1005-1050-1093-1022-1036).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0947__polygon-2237_and_polygon2626__spanned_by_kps(1138-1010-1013-1000-1020)_and(1093-1022-1020-1005-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.0969__polygon-2562_and_polygon2891__spanned_by_kps(1138-1016-1013-1000-1020)_and(1093-1036-1020-1005-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1043__polygon-2839_and_polygon3417__spanned_by_kps(1227-1098-1016-1020-1197)_and(1176-1108-1036-1050-1169).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1203__polygon-2247_and_polygon2153__spanned_by_kps(1302-1098-1010-1016-1020)_and(1213-1108-1022-1036-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1236__polygon-1713_and_polygon2856__spanned_by_kps(1302-1197-1010-1016-1020)_and(1213-1169-1022-1036-1050).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1353__polygon-1204_and_polygon200__spanned_by_kps(1013-1020-1138-1010-1222)_and(1020-1050-1093-1022-1233).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1361__polygon-1953_and_polygon3675__spanned_by_kps(1227-1010-1016-1020-1302)_and(1176-1022-1036-1050-1213).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1381__polygon-694_and_polygon2499__spanned_by_kps(1016-1302-1227-1098-1010)_and(1036-1213-1176-1108-1022).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1398__polygon-1445_and_polygon3279__spanned_by_kps(1020-1302-1227-1098-1010)_and(1050-1213-1176-1108-1022).jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_23h31m35s__04_polygon_matches__02-horiz-ne.jpg___10-alt1-ne.jpg/dist=0.1557__polygon-1120_and_polygon2454__spanned_by_kps(1008-1201-1098-1125-1197)_and(1009-1147-1108-1149-1169).jpg)



#### Matches (inliers) found via StarPolygonMatching

![](docs/example_02_colorized_seams_and_edges/STARS______conf=3.00000___num_inliers=8___02-horiz-ne.jpg___10-alt1-ne.jpg.jpg)

#### Warped images

![](docs/example_02_colorized_seams_and_edges/timelapse_fixed_02-horiz-ne.jpg)

![](docs/example_02_colorized_seams_and_edges/timelapse_fixed_10-alt1-ne.jpg)

#### Seams

![](docs/example_02_colorized_seams_and_edges/timelapse_redseam_fixed_02-horiz-ne.jpg)

![](docs/example_02_colorized_seams_and_edges/timelapse_redseam_fixed_10-alt1-ne.jpg)

#### Colorized edges

![](docs/example_02_colorized_seams_and_edges/timelapse_colorized_edges_fixed_02-horiz-ne.jpg)

![](docs/example_02_colorized_seams_and_edges/timelapse_colorized_edges_fixed_10-alt1-ne.jpg)

#### Assembled panorama

![](docs/example_02_colorized_seams_and_edges/2022-12-30_13h31m14s__fisheye_no-000__red_seams.jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_13h31m48s__fisheye_no-000__colored_edges.jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_13h32m13s__fisheye_multiband-001__colored_edges.jpg)

![](docs/example_02_colorized_seams_and_edges/2022-12-30_12h33m24s__fisheye_multiband-042.jpg)

### Example 3: Removing the waviness effect

![](docs/example_03_waviness/2022-12-30_12h33m29s_no-waviness-correction_cylindrical_multiband-042.jpg)

![](docs/example_03_waviness/2022-12-30_19h50m51s_horizontal-waviness-correction_cylindrical_multiband-042.jpg)

### Example 4: Projections (warp modes)

Plane

![](docs/example_04_warp_modes/2022-12-30_19h58m10s_warp_mode=plane_plane_multiband-042.jpg)

Spherical

![](docs/example_04_warp_modes/2022-12-30_19h58m10s_warp_mode=spherical_spherical_multiband-042.jpg)

Affine

![](docs/example_04_warp_modes/2022-12-30_19h58m11s_warp_mode=affine_affine_multiband-042.jpg)

Cylindrical

![](docs/example_04_warp_modes/2022-12-30_19h58m12s_warp_mode=cylindrical_cylindrical_multiband-042.jpg)

Fisheye

![](docs/example_04_warp_modes/2022-12-30_19h58m12s_warp_mode=fisheye_fisheye_multiband-042.jpg)

Stereographic

![](docs/example_04_warp_modes/2022-12-30_19h58m13s_warp_mode=stereographic_stereographic_multiband-042.jpg)

CompressedPlaneA2B1

![](docs/example_04_warp_modes/2022-12-30_19h58m15s_warp_mode=compressedPlaneA2B1_compressedPlaneA2B1_multiband-042.jpg)

CompressedPlaneA1.5B1

![](docs/example_04_warp_modes/2022-12-30_19h58m16s_warp_mode=compressedPlaneA1.5B1_compressedPlaneA1.5B1_multiband-042.jpg)

Paninia2B1

![](docs/example_04_warp_modes/2022-12-30_19h58m22s_warp_mode=paniniA2B1_paniniA2B1_multiband-042.jpg)

Paninia1.5B1

![](docs/example_04_warp_modes/2022-12-30_19h58m23s_warp_mode=paniniA1.5B1_paniniA1.5B1_multiband-042.jpg)

Mercator

![](docs/example_04_warp_modes/2022-12-30_19h58m29s_warp_mode=mercator_mercator_multiband-042.jpg)

TransverseMercator

![](docs/example_04_warp_modes/2022-12-30_19h58m30s_warp_mode=transverseMercator_transverseMercator_multiband-042.jpg)



### Example 5: Rotating cameras

0°

![](docs/example_05_rotation/2022-12-30_19h47m53s_rot=000deg_fisheye_multiband-042.jpg)

90°

![](docs/example_05_rotation/2022-12-30_19h48m12s_rot=090deg_fisheye_multiband-042.jpg)

...

### Example 6: ORB matching results vs. CustomStarPolygon matching results
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/01-horiz-n.jpg__02-horiz-ne.jpg__ORB__omitted__conf=0.24145__num_inliers=12.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/01-horiz-n.jpg__02-horiz-ne.jpg__STARS__conf=3.00000__num_inliers=9.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/01-horiz-n.jpg__08-horiz-nw.jpg__ORB__omitted__conf=0.55028__num_inliers=29.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/01-horiz-n.jpg__08-horiz-nw.jpg__STARS__conf=3.00000__num_inliers=15.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/01-horiz-n.jpg__09-alt1-n.jpg__ORB__omitted__conf=0.12308__num_inliers=8.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/01-horiz-n.jpg__09-alt1-n.jpg__STARS__conf=3.00000__num_inliers=12.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/02-horiz-ne.jpg__10-alt1-ne.jpg__ORB__omitted__conf=0.15625__num_inliers=8.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/02-horiz-ne.jpg__10-alt1-ne.jpg__STARS__conf=3.00000__num_inliers=9.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/03-horiz-e.jpg__04-horiz-se.jpg__ORB__omitted__conf=0.20522__num_inliers=11.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/03-horiz-e.jpg__04-horiz-se.jpg__STARS__conf=3.00000__num_inliers=12.jpg)




![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/03-horiz-e.jpg__11-alt1-e.jpg__ORB__omitted__conf=0.73227__num_inliers=32.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/03-horiz-e.jpg__11-alt1-e.jpg__STARS__conf=3.00000__num_inliers=8.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/04-horiz-se.jpg__12-alt1-se.jpg__ORB__omitted__conf=0.14799__num_inliers=7.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/04-horiz-se.jpg__12-alt1-se.jpg__STARS__conf=3.00000__num_inliers=11.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/05-horiz-s.jpg__06-horiz-sw.jpg__ORB__omitted__conf=0.83527__num_inliers=36.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/05-horiz-s.jpg__06-horiz-sw.jpg__STARS__conf=3.00000__num_inliers=12.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/05-horiz-s.jpg__13-alt1-s.jpg__ORB__omitted__conf=0.17291__num_inliers=6.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/05-horiz-s.jpg__13-alt1-s.jpg__STARS__conf=3.00000__num_inliers=9.jpg)




![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/06-horiz-sw.jpg__07-horiz-w.jpg__ORB__omitted__conf=0.21480__num_inliers=9.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/06-horiz-sw.jpg__07-horiz-w.jpg__STARS__conf=3.00000__num_inliers=13.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/06-horiz-sw.jpg__14-alt1-sw.jpg__ORB__omitted__conf=0.18868__num_inliers=7.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/06-horiz-sw.jpg__14-alt1-sw.jpg__STARS__conf=3.00000__num_inliers=14.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/07-horiz-w.jpg__08-horiz-nw.jpg__ORB__omitted__conf=0.22959__num_inliers=9.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/07-horiz-w.jpg__08-horiz-nw.jpg__STARS__conf=3.00000__num_inliers=6.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/07-horiz-w.jpg__15-alt1-w.jpg__ORB__omitted__conf=0.16173__num_inliers=6.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/07-horiz-w.jpg__15-alt1-w.jpg__STARS__conf=3.00000__num_inliers=12.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/08-horiz-nw.jpg__16-alt1-nw.jpg__ORB__omitted__conf=0.11070__num_inliers=6.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/08-horiz-nw.jpg__16-alt1-nw.jpg__STARS__conf=3.00000__num_inliers=12.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/11-alt1-e.jpg__12-alt1-se.jpg__ORB__omitted__conf=0.10114__num_inliers=8.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/11-alt1-e.jpg__12-alt1-se.jpg__STARS__conf=3.00000__num_inliers=6.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/12-alt1-se.jpg__13-alt1-s.jpg__ORB__omitted__conf=0.09383__num_inliers=7.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/12-alt1-se.jpg__13-alt1-s.jpg__STARS__conf=3.00000__num_inliers=8.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/13-alt1-s.jpg__14-alt1-sw.jpg__ORB__omitted__conf=0.10249__num_inliers=7.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/13-alt1-s.jpg__14-alt1-sw.jpg__STARS__conf=3.00000__num_inliers=10.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/13-alt1-s.jpg__19-alt2-s.jpg__ORB__omitted__conf=0.11421__num_inliers=9.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/13-alt1-s.jpg__19-alt2-s.jpg__STARS__conf=3.00000__num_inliers=11.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/14-alt1-sw.jpg__15-alt1-w.jpg__ORB__omitted__conf=0.63325__num_inliers=48.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/14-alt1-sw.jpg__15-alt1-w.jpg__STARS__conf=3.00000__num_inliers=9.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/15-alt1-w.jpg__16-alt1-nw.jpg__ORB__omitted__conf=0.08373__num_inliers=7.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/15-alt1-w.jpg__16-alt1-nw.jpg__STARS__conf=3.00000__num_inliers=10.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/17-alt2-n.jpg__18-alt2-e.jpg__ORB__omitted__conf=0.42467__num_inliers=42.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/17-alt2-n.jpg__18-alt2-e.jpg__STARS__conf=3.00000__num_inliers=6.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/17-alt2-n.jpg__20-alt2-w.jpg__ORB__omitted__conf=0.38803__num_inliers=35.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/17-alt2-n.jpg__20-alt2-w.jpg__STARS__conf=3.00000__num_inliers=7.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/18-alt2-e.jpg__19-alt2-s.jpg__ORB__omitted__conf=0.08782__num_inliers=8.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/18-alt2-e.jpg__19-alt2-s.jpg__STARS__conf=3.00000__num_inliers=7.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt1-n.jpg__16-alt1-nw.jpg__ORB__omitted__conf=0.08028__num_inliers=7.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt1-n.jpg__16-alt1-nw.jpg__STARS__conf=3.00000__num_inliers=15.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt1-n.jpg__17-alt2-n.jpg__ORB__omitted__conf=0.08811__num_inliers=8.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt1-n.jpg__17-alt2-n.jpg__STARS__conf=3.00000__num_inliers=11.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt2-s.jpg__20-alt2-w.jpg__ORB__omitted__conf=0.22485__num_inliers=19.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt2-s.jpg__20-alt2-w.jpg__STARS__conf=3.00000__num_inliers=6.jpg)



![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt2-s.jpg__21-zenith.jpg__ORB__omitted__conf=0.23918__num_inliers=21.jpg)
![](docs/example_06_comparison_ORB_matching_vs_StarPolygon_matching/19-alt2-s.jpg__21-zenith.jpg__STARS__conf=3.00000__num_inliers=8.jpg)

## Left to do

- Consider **image distortion**. Take some **chessboard** images and remove image distortion. Does this yield better polygon matching results?
- Add multiprocessing for these (now consecutive) pipeline steps: Star regcognition, star amount reduction by sklearn.cluster.AgglomerativeClustering, polygon measuring
- Drop mistakenly detected stars at the horizon edge.

  ![](docs/left-to-do/stars_horizon.png)

December 2022

Joachim Broser
