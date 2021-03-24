from load_data import load_data
from data_utils import *
import numpy as np
import cv2 as cv
import PIL.Image
from PIL import Image
from IPython.display import clear_output, Image, display
import io


# https://stackoverflow.com/questions/34643747/ipython-notebook-jupyter-opencv-cv2-and-plotting
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))



data_dict = load_data('data/data.p')
velodyne = data_dict['velodyne']
velodyne_r = np.sqrt(np.sum(np.square(velodyne), axis=1))
velodyne_angle = 180 / np.pi * np.arcsin(velodyne[:, 2] / velodyne_r).reshape(-1, 1)


FoV_vertical_min = -24.9
FoV_vertical_max = 2
FoV_vertical = FoV_vertical_max - FoV_vertical_min
angular_res_vert = FoV_vertical / 64
channel = (velodyne_angle - FoV_vertical_min) / angular_res_vert
color = np.mod(channel, 4)
color = np.round(color, 0)

# project the 3d homogeneous points (Lidar coordinates) to 3d homogeneous points (cam0 coordinates)
Xh_velodyne = np.copy(velodyne)
Xh_velodyne[:, 3] = 1

# project from velodyne to cam0
Xh_rectified_cam0 = np.matmul(data_dict['T_cam0_velo'], np.transpose(Xh_velodyne))

# Get rid of points behind cam
indices = np.where(Xh_rectified_cam0[2, :] >= 0)[0]
Xh_rectified_cam0 = Xh_rectified_cam0[:, indices]
color = color[indices]

# project the 3d homogeneous points to 2d homogeneous points on the rectified cam2 image plane
xh_rectified_cam2 = np.matmul(data_dict['P_rect_20'], Xh_rectified_cam0)

# normalize such that homogeneous coordinate is 1.0 again
xh_rectified_cam2 = xh_rectified_cam2 / xh_rectified_cam2[2, :]

# prune homogeneous coordinates
x_rectified_cam2 = xh_rectified_cam2[0:2]

# convert float32 -> int32 (implicit rounding)
x_rectified_cam2 = x_rectified_cam2.astype(np.int32)

#  2nd index goes ove image dimensions
x_rectified_cam2 = np.transpose(x_rectified_cam2)


image_2 = data_dict["image_2"]
image_2_laser_id = np.copy(image_2)

for i in range(np.shape(x_rectified_cam2)[0]):
    x = x_rectified_cam2[i, 0]
    y = x_rectified_cam2[i, 1]

    if color[i] == 0:
        c = (255, 0, 0)
    elif color[i] == 1:
        c = (0, 255, 0)
    elif color[i] == 2:
        c = (0, 0, 255)
    else:
        c = (230, 55, 255)

    cv.circle(img=image_2_laser_id, center=(x, y), radius=1, color=c)

plt.imsave("image_2.png", image_2.astype(np.uint8))
# showarray(image_2)
plt.imsave("image_2_laser_id.png", image_2_laser_id.astype(np.uint8))
# showarray(image_2_laser_id)
plt.imshow(image_2)
plt.imshow(image_2_laser_id)
plt.show()