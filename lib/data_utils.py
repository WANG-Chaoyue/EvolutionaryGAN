import numpy as np
from sklearn import utils as skutils
from PIL import Image
from rng import np_rng, py_rng

def center_crop(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = int(round((h - ph)/2.))
    i = int(round((w - pw)/2.))
    return x[j:j+ph, i:i+pw]

def patch(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = py_rng.randint(0, h-ph)
    i = py_rng.randint(0, w-pw)
    x = x[j:j+ph, i:i+pw]
    return x

def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], basestring):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])

def convert_img(img):
    H,W,C=img.shape
    con_img = np.zeros((C,H,W),dtype=img.dtype)
    for i in xrange(C):
        con_img[i,:,:] = img[:,:,i]
    return con_img

def convert_img_back(img):
    C,H,W=img.shape
    con_img = np.zeros((H,W,C),dtype=img.dtype)
    for i in xrange(C):
        con_img[:,:,i] = img[i,:,:]
    return con_img

def convert_video(video):
    n,f,c,w,h = video.shape
    con_video = np.zeros((n,c,f,w,h),dtype='float32')
    for i in xrange(f):
        con_video[:,:,i,:,:] = video[:,i,:,:,:]
    return con_video

def convert_video_back(video):
    n,c,f,w,h = video.shape
    con_video = np.zeros((n,f,c,w,h),dtype='float32')
    for i in xrange(f):
        con_video[:,i,:,:,:] = video[:,:,i,:,:]
    return con_video

def coloring(xmb, ymb, p=0.2):
    xmb_color = xmb 
    B,C,H,W = xmb.shape 
    #print H
    for i in range(B):	
        img_input = np.cast['float32'](xmb[i,:,:,:])
	img_real = np.cast['float32'](ymb[i,:,:,:])
        
        img_input = convert_img_back(img_input)
        img_real = convert_img_back(img_real)
	
	img_gray = img_real[:,:,0] + img_real[:,:,1] + img_real[:,:,2] 
        img_gray = (765 - img_gray)
        img_gray[img_gray>10]=255 
        img_gray[img_gray<255]=0
        img_gray = img_gray/255
        img_gray = [img_gray]*3
        img_gray = np.array(img_gray)
	img_gray = img_gray.reshape([3,H,W])
        img_mask = convert_img_back(img_gray)
        rand = np.random.random((H,W))
        rand[rand>p]=1
        rand[rand<1]=0
        rand = [rand]*3
        rand = np.array(rand)
        rand = rand.reshape([3,H,W])
        rand = convert_img_back(rand)
 
        img_color = (img_input) * (1-img_mask) + img_real * img_mask + rand*1000
        img_color[img_color>255]=255 
        img_input = img_color*(img_input/255)
        img_input = np.cast['uint8'](img_input)
        # img_real = np.cast['uint8'](img_real)
        # img_mask = np.cast['uint8'](img_mask*255)
        xmb_color[i,:,:,:] = convert_img(img_input)
    return xmb_color

def processing_img(img, center=True, scale=True, convert=True):
    img = np.array(img)
    img = np.cast['float32'](img)
    if convert is True:
        img = convert_img(img)
    if  center and scale:
        img[:] -= 127.5
        img[:] /= 127.5
    elif center:
        img[:] -= 127.5
    elif scale:
        img[:] /= 255.

    return img 

def ImgRescale(img,center=True,scale=True, convert_back=False):
    img = np.array(img)
    img = np.cast['float32'](img)
    if convert_back is True:
        img = convert_img_back(img)
    if center and scale:
        img = ((img+1) / 2 * 255).astype(np.uint8) 
    elif center:
	img = (img + 127.5).astype(np.uint8) 
    elif scale:
	img = (img * 255).astype(np.uint8) 
    return img

def ImgBatchRescale(img,center=True,scale=True, convert_back=False):
    img = np.array(img)
    img = np.cast['float32'](img)
    if convert_back is True:
        b,C,H,W = img.shape
        print img.dtype
        imgh = np.zeros((b,H,W,C),dtype=img.dtype)
        for i in range(b):
            imgh[i,:,:,:] = convert_img_back(img[i,:,:,:])
        img = imgh
    if center and scale:
        img = ((img+1) / 2 * 255).astype(np.uint8) 
    elif center:
	img = (img + 127.5).astype(np.uint8) 
    elif scale:
	img = (img * 255).astype(np.uint8) 
    return img

def test_color(img, p=0.2):
    img = np.array(img)
    img_color = img 
    H,W,C = img.shape 
    #print C
    img = np.cast['float32'](img)
    img_color = np.cast['float32'](img_color)
	
    img = img[:,:,0] + img[:,:,1] + img[:,:,2] 
    img[img>100]=255
    img[img<255]=0
    img = img/255
    img = [img]*3
    img = np.array(img)
    img = img.reshape([C,H,W])
    img = convert_img_back(img)
    rand = np.random.random((H,W))
    rand[rand>p]=1
    rand[rand<1]=0
    rand = [rand]*3
    rand = np.array(rand)
    rand = rand.reshape([C,H,W])
    rand = convert_img_back(rand)
 
    img_color = img_color + rand*1000
    img_color[img_color>255]=255 
    img_color = img_color*img
    img_color = np.cast['uint8'](img_color)
    
    return img_color

def CropImgs(xmb, ymb, fineSize, nbatch, h1, w1):
    if nbatch == 1:
        xmb_out = xmb[:,:,h1[0]:h1[0]+fineSize,w1[0]:w1[0]+fineSize] 
	ymb_out = ymb[:,:,h1[0]:h1[0]+fineSize,w1[0]:w1[0]+fineSize]
    else:
        xmb_out = np.zeros((nbatch,3,fineSize,fineSize),dtype='uint8')
        ymb_out = np.zeros((nbatch,3,fineSize,fineSize),dtype='uint8')
        for i in xrange(nbatch):
            x_img = xmb[i,:,:,:]
            y_img = ymb[i,:,:,:]
            h = h1[i]
            w = w1[i]
	    x_img_out = x_img[:,h:h+fineSize,w:w+fineSize]
	    y_img_out = y_img[:,h:h+fineSize,w:w+fineSize]
	    xmb_out[i,:,:,:] = x_img_out 
	    ymb_out[i,:,:,:] = y_img_out 
    xmb_out = xmb_out.reshape([nbatch,3,fineSize,fineSize]) 
    ymb_out = ymb_out.reshape([nbatch,3,fineSize,fineSize]) 
    return xmb_out, ymb_out

def FlipImgs(xmb, ymb):
    bs, ci, hi, wi = xmb.shape	
    xmb_out = np.zeros((bs,ci,hi,wi),dtype='uint8')
    ymb_out = np.zeros((bs,ci,hi,wi),dtype='uint8')
    for i in xrange(bs):
        x_img = xmb[i,:,:,:]
        y_img = ymb[i,:,:,:]
        if np.random.rand(1) > 0.5:
            x_img = convert_img_back(x_img)
            y_img = convert_img_back(y_img)

            x_img = Image.fromarray(x_img)	
            y_img = Image.fromarray(y_img)	
		
            x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
            y_img = y_img.transpose(Image.FLIP_LEFT_RIGHT)
		
            x_img = convert_img(np.array(x_img))
            y_img = convert_img(np.array(y_img))
	xmb_out[i,:,:,:] = x_img 
	ymb_out[i,:,:,:] = y_img 
    xmb_out = xmb_out.reshape([bs,ci,hi,wi]) 
    ymb_out = ymb_out.reshape([bs,ci,hi,wi]) 
    return xmb_out, ymb_out

def pix2pixBatch(xmb,ymb,fineSize,input_nc,flip=True):
    bs, ci, hi, wi = xmb.shape	
    if hi > fineSize or wi > fineSize: 
        h1 = np.random.randint(0,hi-fineSize,size=bs) 
        w1 = np.random.randint(0,wi-fineSize,size=bs)
	xmb,ymb = CropImgs(xmb, ymb, fineSize, bs, h1, w1)
    else:
        xmb = xmb.reshape([bs,3,fineSize,fineSize])
        ymb = ymb.reshape([bs,3,fineSize,fineSize])
    if flip is True:
	xmb,ymb = FlipImgs(xmb,ymb)
    
    return xmb, ymb

def Batch(xmb,fineSize,input_nc,flip=True):
    bs, ci, hi, wi = xmb.shape	
    if hi > fineSize or wi > fineSize: 
        h1 = np.random.randint(0,hi-fineSize,size=bs) 
        w1 = np.random.randint(0,wi-fineSize,size=bs)
	xmb,ymb = CropImgs(xmb, xmb, fineSize, bs, h1, w1)
    else:
        xmb = np.array(xmb,dtype='uint8')
        xmb = xmb.reshape([bs,3,fineSize,fineSize])
        ymb = xmb
    if flip is True:
	xmb,ymb = FlipImgs(xmb,ymb)
    
    return xmb
