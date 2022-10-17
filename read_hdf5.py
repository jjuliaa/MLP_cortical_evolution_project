import numpy as np
import h5py
import pylab as pl
import matplotlib.image as mpimg
from scipy.interpolate import interp1d
# Read original images
img = mpimg.imread('trainimg_H_x9-01.png')
img2 = mpimg.imread('trainimg_Q_x9-01.png')


x = h5py.File('/Users/julialc4/Desktop/MLP/b2/logs/out.h5', 'r')

Error = x['errors'][:] #get everything inside the labelled h5 key'/error'

print("E: ", Error)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#first image
OutNode1 = x['output1'][:]
OutNode2 = x['output2'][:]
OutNode3 = x['output3'][:]

x1 = x['x1'][:]
y1 = x['y1'][:]

#second image
OutNodeA1 = x['outputA1'][:]
OutNodeA2 = x['outputA2'][:]
OutNodeA3 = x['outputA3'][:]

x2 = x['x2'][:]
y2 = x['y2'][:]

#scaling question
print("x1 shape", x1.shape)
print("x2 shape", x2.shape)


Y = np.ones([int(np.max(x1))+1, int(np.max(y1))+1, 3])

for i in range(len(x1)):
    Y[int(x1[i]), int(y1[i]), 2] = OutNode1[i]> 0.5
    Y[int(x1[i]), int(y1[i]), 0] = OutNode2[i]>0.5
    Y[int(x1[i]), int(y1[i]), 1] = OutNode3[i]>0.5 #filters out otherwise is raw

print(Y.shape)

Y = np.transpose(Y, [1,0,2]) #swap x, y
print(Y.shape)


Y2 = np.ones([int(np.max(x2))+1, int(np.max(y2))+1, 3])

for i in range(len(x2)):
    Y2[int(x2[i]), int(y2[i]), 2] = OutNodeA1[i]> 0.5
    Y2[int(x2[i]), int(y2[i]), 0] = OutNodeA2[i]>0.5
    Y2[int(x2[i]), int(y2[i]), 1] = OutNodeA3[i]>0.5 #filters out otherwise is raw

Y2 = np.transpose(Y2, [1,0,2]) #swap x, y

#interp image
OutNodeB1 = x['outputB1'][:]
OutNodeB2 = x['outputB2'][:]
OutNodeB3 = x['outputB3'][:]


print("x1:", x1)
print("x2:", x2)
#x3 = np.ones(int(np.max(x1))+1)
#y3 = np.ones(int(np.max(y1))+1)

#How to do this?
x3 = x1
y3 = y1

Y3 = np.ones([int(np.max(x3))+1, int(np.max(y3))+1, 3])
for i in range(len(x3)):
    Y3[int(x3[i]), int(y3[i]), 2] = OutNodeB1[i]> 0.5
    Y3[int(x3[i]), int(y3[i]), 0] = OutNodeB2[i]>0.5
    Y3[int(x3[i]), int(y3[i]), 1] = OutNodeB3[i]>0.5 #filters out otherwise is raw

Y3 = np.transpose(Y3, [1,0,2]) #swap x, y



#interp image progeny
OutNodeC1 = x['outputC1'][:]
OutNodeC2 = x['outputC2'][:]
OutNodeC3 = x['outputC3'][:]

#How to do this?
x4 = x1
y4 = y1

Y4 = np.ones([int(np.max(x4))+1, int(np.max(y4))+1, 3])
for i in range(len(x3)):
    Y4[int(x4[i]), int(y4[i]), 2] = OutNodeC1[i]> 0.5
    Y4[int(x4[i]), int(y4[i]), 0] = OutNodeC2[i]>0.5
    Y4[int(x4[i]), int(y4[i]), 1] = OutNodeC3[i]>0.5 #filters out otherwise is raw

Y4 = np.transpose(Y4, [1,0,2]) #swap x, y

F = pl.figure()
f = F.add_subplot(241)
f.set_xlabel('Error \n T: 10000000, nJ: 15, nQ: 15, lRate: 0.03')
f.plot(smooth(Error, 3333))
f = F.add_subplot(243)
f.set_xlabel('1st map')
f.imshow(Y)
f = F.add_subplot(244)
f.set_xlabel('1st map original')
f.imshow(img)
f = F.add_subplot(245)
f.set_xlabel('2nd map')
f.imshow(Y2)
f = F.add_subplot(246)
f.set_xlabel('2nd map original')
f.imshow(img2)
f = F.add_subplot(247)
f.set_xlabel('interp map: ancestor (x1, y1)')
f.imshow(Y3)
f = F.add_subplot(248)
f.set_xlabel('interp map: progeny (x1, y1) ')
f.imshow(Y4)

pl.show()


