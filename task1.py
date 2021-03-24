import os
from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt

#Loading the dictionary
data_path = os.path.join('data/','data.p')
data = load_data(data_path)

#Taking the required data from the dictionary
velodyne = data.get('velodyne')
X = velodyne[:,0]
Y = velodyne[:,1]
alpha = velodyne[:,3]
xyz = np.zeros((np.size(X),3))
xyz[:,0] = X
xyz[:,1] = Y
xyz[:,2] = alpha

#Finding the min, max of the coordinates and the no. of pixels to be used later for temperory arrays
X_max = np.max(X)
Y_max = np.max(Y)
X_min = np.min(X)
Y_min = np.min(Y)
no_pix_x = int((X_max - X_min)/0.2)
no_pix_y = int((Y_max - Y_min)/0.2)

#Creating a temperary array to create the required resolution grid and filter coinciding points in the same grid
l = np.size(X)
xyz_filt1 = np.zeros((no_pix_x*no_pix_y,3))
a = 0
while a < l:
    n = int((xyz[a,0]- X_min)/0.2)
    m = int((xyz[a,1]- Y_min)/0.2)
    if xyz[a,2] >= xyz_filt1[n*m,2]:
        xyz_filt1[n*m,:] = xyz[a,:]
    a = a + 1

#Removing the rows with no data. Second filtering to get final array
a = 0
b = 0
xyz_filt2 = np.zeros((np.size(X),3))
while a < len(xyz_filt1):
    if xyz_filt1[a,0] != 0 or xyz_filt1[a,1] != 0:
        xyz_filt2[b,:] = xyz_filt1[a,:]
        b = b + 1
    a = a + 1

#Plot using scatter plot with black background 
plt.style.use('dark_background')
plt.scatter(xyz_filt2[:,0], xyz_filt2[:,1],  c = xyz_filt2[:,2], s = 0.02, cmap = 'Greys')
plt.show()

#End of code