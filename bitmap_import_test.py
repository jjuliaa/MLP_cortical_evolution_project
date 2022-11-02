# Read and code colors from bitmaps
import numpy as np
import h5py
import matplotlib.image as mpimg



def color_code(img):
    X = np.zeros([1,3])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            rgb = np.array(img[y,x,:3])
            #print("colors", rgb)
            if not((rgb[0] == 0) and (rgb[1] == 0) and (rgb[2] == 0)):
                if((rgb[0]==0) and (rgb[1]==0) and (rgb[2]==1)):
                    X = np.vstack([X,[x,y,1]]) # blue is 1, v1
                elif ((rgb[0] == 1) and (rgb[1] == 0) and (rgb[2] == 0)):
                    X = np.vstack([X,[x,y,2]]) # red is 2 s1
                elif ((rgb[0] == 0) and (rgb[1] == 1) and (rgb[2] == 0)):
                    X = np.vstack([X,[x, y, 3]]) # green is 3 a1
                else:
                    X = np.vstack([X,[x, y, 0]]) # eveything else (including white) is 0
    return X[1:,:]


# Read Images
img = mpimg.imread('trainimg_H_x9-01.png')
#img2 = mpimg.imread('trainimg_Q_x9-01.png')
img2 = mpimg.imread('trainimg_H8_RM_x9-01.png')
img3 = mpimg.imread('trainimg_H8_RM_x9-01.png')
print("----img shapes:----")
print(img.shape)
print(img2.shape)
print(img3.shape)
print("----done.----")


#scale images to have the same number of pixels/dimensions (needs to be addressed)

#filtering colors from floats to integers
img = (img>0.5)*1
img2 = (img2>0.5)*1
img3 = (img3>0.5)*1

imgcoded = color_code(img)
img2coded = color_code(img2)
img3coded = color_code(img3)

np.savez('img1.npz', X=imgcoded)
np.savez('img2.npz', X=img2coded)
np.savez('img3.npz', X=img3coded)
#np.savez('img2.npz', X=img2coded)

datax1 = imgcoded[0]
x1 = imgcoded[:,0]
y1 = imgcoded[:,1]
c1 = imgcoded[:,2]

datax2 = img2coded[0]
x2 = img2coded[:,0]
y2 = img2coded[:,1]
c2 = img2coded[:,2]

datax3 = img3coded[0]
x3 = img3coded[:,0]
y3 = img3coded[:,1]
c3 = img3coded[:,2]

#interpolated points
#for i in range(len(x1)):
#    y3_interp1 = np.interp(i, x1[i], x2[i])

h51 = h5py.File('/Users/julialc4/Desktop/MLP/b2/img1coded.h5','w')
h51.create_dataset('x1', data= x1 *1.0)
h51.create_dataset('y1', data= y1 *1.0)
h51.create_dataset('c1', data= c1 *1.0)
h51.close()

h52 = h5py.File('/Users/julialc4/Desktop/MLP/b2/img2coded.h5','w')
h52.create_dataset('x2', data= x2 *1.0)
h52.create_dataset('y2', data= y2 *1.0)
h52.create_dataset('c2', data= c2 *1.0)
h52.close()

h53 = h5py.File('/Users/julialc4/Desktop/MLP/b2/img3coded.h5','w')
h53.create_dataset('x3', data= x2 *1.0)
h53.create_dataset('y3', data= y2 *1.0)
h53.create_dataset('c3', data= c2 *1.0)
h53.close()

print("..Finished creating h5s")
print()

#for hdf5 loading stuff, this can be done directly in c++
'''
f = h5py.File('mydataset.hdf5', 'a')
grp = f.create_group("subgroup")

dset2 = grp.create_dataset("another_dataset", (50,), dtype='f')

dset2.name

dset3 = f.create_dataset('subgroup2/dataset_three', (10,), dtype='i')

dset3.name
'/subgroup2/dataset_three'

'/subgroup/another_dataset'

'''
