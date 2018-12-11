import os, sys
sys.path.append('..')

import numpy as np
import h5py
import random 
from matplotlib.pyplot import imshow, imsave
from PIL import Image

from lib.data_utils import convert_img, convert_img_back

def toy_dataset(DATASET='8gaussians', size=256):

    if DATASET == '25gaussians':
        dataset = []
        for i in xrange(500/25):
            for x in xrange(-2, 3):
                for y in xrange(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
        #while True:
        #    for i in xrange(len(dataset)/BATCH_SIZE):
        #        yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':
        data = sklearn.datasets.make_swiss_roll(
            n_samples=size, 
            noise=0.25
        )[0]
        dataset = data.astype('float32')[:, [0, 2]]
        dataset /= 7.5 # stdev plus a little

    elif DATASET == '8gaussians':
    
        scale = 2.
        centers = [
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(scale*x,scale*y) for x,y in centers]
        dataset = []
        for i in xrange(size):
            point = np.random.randn(2)*.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414 # stdev

    return dataset 



def faces_beds(data_path='~/Datasets/CelebA/img_align_celeba_crop/',
          img_shape=[3,150,150], save = False,
          start=0, stop=200):
    C, H, W = img_shape
    files_name = os.listdir(data_path)

    X = np.zeros(((stop-start), C, H, W), dtype='uint8')
    for i in xrange(start,stop):
        z = 0
        img = Image.open(data_path + files_name[i])
	
	if img.size[1] != H or img.size[0] != W: 
	    img = img.resize((W, H),Image.ANTIALIAS)
	
        img = np.array(img)
	if len(np.shape(img)) < C:
            img = [img]*C
	    img = np.array(img)
	    img = img.reshape([C,H,W*2])
	    img = convert_img_back(img) 
		
	if i == 0 and save is True:
            imsave('A_%s'%(files_name[i]),img,format='jpg')
        
	img = convert_img(img)

	X[(i-start),:,:,:] = img
        
    return X, files_name[start:stop]


if __name__ == '__main__':
    start = 0
    stop  = 202599 
    loadSize = 64 
    train_data = '~/CelebA/img_align_celeba_crop/'
    tra_X, _ = faces_beds(data_path=train_data, img_shape=[3,loadSize,loadSize],
		          save = True, start=start, stop=stop)
    
    f = h5py.File('~/Datasets/CelebA/img_align_celeba_64.hdf5','w')
    dset = f.create_dataset("data",data=tra_X)

