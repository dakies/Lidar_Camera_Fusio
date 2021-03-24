import os
from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt

#Loading data
data_path = os.path.join('data/','data.p')
data = load_data(data_path)

#Getting the required data from the dictionary into seperate data sets
velodyne = data.get('velodyne')
cam2_image = data.get('image_2')
p_r20 = data.get('P_rect_20')
t_c2v = data.get('T_cam0_velo')
sem_label = data.get('sem_label')
color_map = data.get('color_map')
bbox = data.get('objects')

#Filtering the velodyne data to get only the front part of the scan
X = velodyne[:,0]
a = 0
b = 0
velodyne_filt = np.zeros((np.size(X),4))
c2_color = np.zeros((np.size(X),3))

while a < np.size(X):
    if velodyne[a,0] > 0:
        velodyne_filt[b,:] = velodyne[a,:]
        c2_color[b,:] = color_map.get(sem_label[a,0])
        b = b + 1
    a = a + 1

velodyne_filt1 = velodyne_filt[:b,:]
c2_color1 = c2_color[:b,:]
velodyne_filt1[:,3] = 1

#Transforming the point cloud to camera coordinates
c0_points = np.dot(t_c2v,velodyne_filt1.T)
c2_points = np.dot(p_r20,c0_points).T
norm = c2_points[:,2:]
c2_points = c2_points/norm

#Changing BGR to RGB
R_c = c2_color1[:,2]
G_c = c2_color1[:,1]
B_c = c2_color1[:,0]
c2_color1[:,0] = R_c
c2_color1[:,1] = G_c
c2_color1[:,2] = B_c

#To get the edge points in the camera 2 coordinates after correcting the rotation
rot_y = np.zeros((3,3))
edge = np.zeros((len(bbox),8,4))
a = 0
while a < len(bbox):
    c = bbox[a]
    x = c[11]
    y = c[12]
    z = c[13]
    h = c[8]
    l = c[9]
    w = c[10]
    r = c[14]
    rot_y[0,0] = np.cos(r)
    rot_y[0,2] = np.sin(r)
    rot_y[1,1] = 1
    rot_y[2,0] = -1*np.sin(r)
    rot_y[2,2] = np.cos(r)
    check = np.dot(rot_y,np.array([w/2,-h,l/2])) + np.array([x,y,z])
    edge[a,0,:3] = np.dot(rot_y,np.array([w/2,-h,l/2])) + np.array([x,y,z])
    edge[a,1,:3] = np.dot(rot_y,np.array([-w/2,-h,l/2])) + np.array([x,y,z])
    edge[a,2,:3] = np.dot(rot_y,np.array([-w/2,-h,-l/2])) + np.array([x,y,z])
    edge[a,3,:3] = np.dot(rot_y,np.array([w/2,-h,-l/2])) + np.array([x,y,z])
    edge[a,4,:3] = np.dot(rot_y,np.array([w/2,0,l/2])) + np.array([x,y,z])
    edge[a,5,:3] = np.dot(rot_y,np.array([-w/2,0,l/2])) + np.array([x,y,z])
    edge[a,6,:3] = np.dot(rot_y,np.array([-w/2,0,-l/2])) + np.array([x,y,z])
    edge[a,7,:3] = np.dot(rot_y,np.array([w/2,0,-l/2])) + np.array([x,y,z])
    a = a + 1

edge[:,:,3] = 1

#Defining the connect vector used to create the bounding box
connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])


#Plotting the coloured point cloud, the bounding box and the camera 2 image in the same plot
plt.scatter(c2_points[:b,0],c2_points[:b,1],c = c2_color1[:b,:]/255, s = 0.2)
b = 0
while b < len(bbox):
    a = 0
    ed_points = np.dot(p_r20,edge[b,:,:].T).T
    norm1 = ed_points[:,2:]
    ed_points = ed_points/norm1
    while a < np.size(connect[:,0]):
        plt.plot([ed_points[connect[a,0],0],ed_points[connect[a,1],0]],[ed_points[connect[a,0],1],ed_points[connect[a,1],1]], color = 'green', linewidth = 2)
        a = a + 1
    b = b + 1
plt.imshow(cam2_image)
plt.show()

#End of code
