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


cam2cam = calib_cam2cam('data/problem_4/calib_cam_to_cam.txt', mode='02')
velo2cam = calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')

# transformation matrix from velodyne to cam0 3d
T = np.append(velo2cam[0], velo2cam[1], axis=1)
T = np.append(T, [[0, 0, 0, 1]], axis=0)

imu2velo = calib_velo2cam('data/problem_4/calib_imu_to_velo.txt')
imu2velo = np.append(imu2velo[0], imu2velo[1], axis=1)
imu2velo = np.append(imu2velo, [[0, 0, 0, 1]], axis=0)

velo2imu = np.linalg.inv(imu2velo)


def transform2cam(point_cloud, T):
    # cam0 3d coordinates
    Xh_cam0 = np.matmul(T, point_cloud.T)

    # here eliminate points in the back
    indices = np.where(Xh_cam0[2, :] > 0)[0]
    Xh_cam0 = Xh_cam0[:, indices]

    # cam2 to image plane
    cam2coord2d = np.matmul(cam2cam, Xh_cam0)
    cam2coord2d = cam2coord2d / cam2coord2d[2, :]
    cam2coord2d = cam2coord2d[:2, :]

    distances = np.sqrt(np.sum(np.square(Xh_cam0), axis=0))
    colormap = depth_color(distances)

    return cam2coord2d, colormap


def removeMotionDistortion(index):
    # read all the required data
    image = cv.imread('data/problem_4/image_02/data/' + index + '.png', )
    point_cloud = load_from_bin('data/problem_4/velodyne_points/data/' + index + '.bin')[:, :4]
    lidar_t_start = compute_timestamps('data/problem_4/velodyne_points/timestamps_start.txt', index)
    lidar_t_trigger = compute_timestamps('data/problem_4/velodyne_points/timestamps.txt', index)
    # lidar_t_trigger = compute_timestamps('data/problem_4/image_02/timestamps.txt', index)
    lidar_t_end = compute_timestamps('data/problem_4/velodyne_points/timestamps_end.txt', index)
    v = load_oxts_velocity('data/problem_4/oxts/data/' + index + '.txt')
    angular_rate = np.asarray(load_oxts_angular_rate('data/problem_4/oxts/data/' + index + '.txt'))
    angular_rate_z = angular_rate[2]

    # Angle of LiDAR sensor with respect to starting position (clockwise)
    trigger_angle = (lidar_t_trigger - lidar_t_start) / (lidar_t_end - lidar_t_start) * 360
    # Always 180. LiDAR starts from behind and rotates cw.

    # compute angle for each point in degree
    angles = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
    angles = (180 / np.pi * angles)

    # Due to properties of arctan2 there are negative values on right, which we convert to positive
    indices = np.where(angles < 0)[0]
    angles[indices] += 360

    # Angles in rad with respect to camera trigger. Add and modulo are for the possibility of having angle = 360
    angles_relative = (angles + 360 - trigger_angle) % 360
    angles_relative = np.pi * angles_relative / 180


    # time stamp of individual lasers
    timestamp_laser = lidar_t_start + (lidar_t_end - lidar_t_start) * (angles_relative / (2 * np.pi))
    timestamp_laser2trigger = timestamp_laser - lidar_t_trigger

    # Convert to homogeneous coordinates
    point_cloud = np.append(point_cloud, np.ones((len(point_cloud), 1)), axis=1)
    point_cloud_original = np.copy(point_cloud)

    # Convert to IMU coordinates
    point_cloud = np.matmul(velo2imu, point_cloud.T).T

    # now iterate over the points and correct them
    for i in range(len(point_cloud)):
        translation = timestamp_laser2trigger[i] * v
        # translation = np.zeros(v.shape)
        angle = -timestamp_laser2trigger[i] * angular_rate_z
        R = np.array([[np.cos(-angle), -np.sin(-angle), 0],
                      [np.sin(-angle), np.cos(-angle), 0],
                      [0, 0, 1]
                      ])
        # R = np.identity(3)
        M = np.append(R, -translation.reshape(-1, 1), axis=1)
        M = np.append(M, [[0, 0, 0, 1]], axis=0)

        point_cloud[i, :] = np.matmul(M, point_cloud[i, :].T).T

    # Transform back to Lidar coordinates
    point_cloud = np.matmul(imu2velo, point_cloud.T).T

    # Project to cam coordinates
    cam2coord2d, colormap = transform2cam(point_cloud, T)
    cam2coord2d_distortion, colormap_distortion = transform2cam(point_cloud_original, T)

    image2 = np.copy(image)
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
    plt.imshow(image2)

    image_distortion = print_projection_plt(cam2coord2d_distortion, colormap_distortion, image.astype(np.uint8))
    plt.imsave("image_" + index + "distortion" + ".png", image_distortion)
    plt.imshow(image_distortion)
    # showarray(image_distortion)
    plt.show()

    image_distortion_removed = print_projection_plt(cam2coord2d, colormap, image)
    plt.imsave("image_" + index + "distortion_corrected" + ".png", image_distortion_removed.astype(np.uint8))
    plt.imshow(image_distortion_removed)
    # showarray(image_distortion_removed)
    plt.show()

# %%

indecies = ['0000000037', '0000000312', '0000000340']
for i in range(3):
    removeMotionDistortion(indecies[i])