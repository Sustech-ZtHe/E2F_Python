import os
import shutil

work_path = os.getcwd()
h3d_road = os.path.join(work_path, "hough-3d-lines-master")
# origin file road
origin_h3d_main = f"{h3d_road}/hough3dlines.cpp"
origin_pointcloud = f"{h3d_road}/pointcloud.cpp"
origin_vector3d = f"{h3d_road}/vector3d.cpp"
origin_vector3d_h = f"{h3d_road}/vector3d.h"
# replace file road
replace_h3d_main = f"{work_path}/hough_replace/hough3dlines_replaced.cpp"
replace_pointcloud = f"{work_path}/hough_replace/pointcloud_replaced.cpp"
replace_vector3d = f"{work_path}/hough_replace/vector3d_replaced.cpp"
replace_vector3d_h = f"{work_path}/hough_replace/vector3d_replaced.h"

# replace file
shutil.copyfile(replace_h3d_main, origin_h3d_main)
shutil.copyfile(replace_pointcloud, origin_pointcloud)
shutil.copyfile(replace_vector3d, origin_vector3d)
shutil.copyfile(replace_vector3d_h, origin_vector3d_h)