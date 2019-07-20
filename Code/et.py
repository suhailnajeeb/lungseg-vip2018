import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
import sys

# Function to distort scan and mask
def elastic_transform(image, mask, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    
    random_state = np.random.RandomState(random_state)
    
    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size,center_square[1] - square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine,alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    print(np.min(image))
    image_w = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=int(np.min(image)))
    mask_w = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)
    
    blur_size = int(2*sigma) | 1
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    
    grid_x = (grid_x + dx).astype(np.float32)
    grid_y = (grid_y + dy).astype(np.float32)
    
    image_d = cv2.remap(image_w, grid_x, grid_y, interpolation=cv2.INTER_LINEAR)
    mask_d = cv2.remap(mask_w, grid_x, grid_y, interpolation=cv2.INTER_LINEAR)
    
    return image_d, mask_d

if __name__ == '__main__': 

    dbPath = '/media/shahruk/Terra 2.0D/VIP Cup 2018/dbHdf5/dataset1_2d_onlyTumor_cropped_x-75-425_y-75-425.hdf5'

    db = h5py.File(dbPath, 'r')
    X = db['slice'][...]
    X = np.float32(X)
    X = np.expand_dims(X, -1)

    Y = db['mask'][...]
    Y = np.expand_dims(Y, -1)
    Y = np.float32(Y)


    x = X[int(sys.argv[1]), ...]
    y = Y[int(sys.argv[1]), ...]

    xd,yd = elastic_transform(x,y, x.shape[1] * 0.15, x.shape[1] * 0.12, 1234)
    # x2 = b[...,0]

    w = x.shape[0]
    h = x.shape[1]

    plt.subplot(3, 2, 1)
    plt.imshow(x.reshape(w,h), cmap='gray')
    print(np.min(x))
    plt.subplot(3, 2, 2)
    plt.imshow(xd.reshape(w,h), cmap='gray')

    plt.subplot(3, 2, 3)
    plt.imshow(y.reshape(w,h), cmap='gray')

    plt.subplot(3, 2, 4)
    plt.imshow(yd.reshape(w,h), cmap='gray')

    plt.subplot(3, 2, 5)
    plt.imshow(np.multiply(y.reshape(w,h), x.reshape(w,h)), cmap='gray')

    plt.subplot(3, 2, 6)
    plt.imshow( np.multiply(yd.reshape(w,h), xd.reshape(w,h)), cmap='gray')

    plt.show()

    print np.mean(abs(yd-y))
