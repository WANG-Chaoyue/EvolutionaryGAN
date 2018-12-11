import sys
sys.path.append('..')

import os
import scipy.io as scio
import numpy as np
import theano
import theano.tensor as T
import lasagne
import h5py 
import shutil
import json
import random

from time import time
from PIL import Image
from lib.data_utils import processing_img, convert_img_back, convert_img, Batch, shuffle, iter_data, ImgRescale, OneHot
from lib.theano_utils import floatX, sharedX
from lib.rng import py_rng, np_rng, t_rng
from models import models_uncond 

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def create_G(loss_type=None, discriminator=None, lr=0.0002, b1=0.5, ngf=64): 
    noise = T.matrix('noise')
    generator = models_uncond.build_generator_128(noise,ngf=ngf)
    Tgimgs = lasagne.layers.get_output(generator)
    Tfake_out = lasagne.layers.get_output(discriminator, Tgimgs)
    
    if loss_type == 'trickLogD':
        generator_loss = lasagne.objectives.binary_crossentropy(Tfake_out, 1).mean()
    elif loss_type == 'minimax': 
        generator_loss = -lasagne.objectives.binary_crossentropy(Tfake_out, 0).mean()
    elif loss_type == 'ls': 
        generator_loss = T.mean(T.sqr((Tfake_out - 1)))
    
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    updates_g = lasagne.updates.adam(generator_loss, generator_params, learning_rate=lr, beta1=b1)
    train_g = theano.function([noise],
                              generator_loss,
                              updates=updates_g)
    gen_fn = theano.function([noise],
                         lasagne.layers.get_output(generator,
                              deterministic=True))
    return train_g, gen_fn, generator 

def main():
    # Parameters
    data_path = '../datasets/'
    task = 'face'
    name = '128'

    start = 0
    stop = 202560 
    input_nc = 3 
    loss_type = ['trickLogD','minimax','ls'] 
    nloss = 3
    shuffle_ = True 
    batchSize = 32 
    fineSize = 128 
    flip = True
   
    ncandi = 1         # # of survived childern 
    kD = 3             # # of discrim updates for each gen update
    kG = 1             # # of discrim updates for each gen update
    ntf = batchSize*kD 
    b1 = 0.5           # momentum term of adam
    nz = 100           # # of dim for Z
    ngf = 64           # # of gen filters in first conv layer
    ndf = 64           # # of discrim filters in first conv layer
    niter = 25         # # of iter at starting learning rate
    lr = 0.0002        # initial learning rate for adam G
    lrd = 0.0002       # initial learning rate for adam D
    beta = 0.001       # the hyperparameter that balance fitness score 
    GP_norm = False    # if use gradients penalty on discriminator
    LAMBDA = 2.        # hyperparameter of GP 
    
    save_freq = 5000 
    show_freq = 500 
    begin_save = 0 
    test_deterministic = True   
    
    # Load the dataset
    print("Loading data...")
    f = h5py.File(data_path+'img_align_celeba_128.hdf5','r') 
    trX = f['data'] 
    ids = range(start, stop)
    
    ################## MODEL D ####################### 
    print("Building model and compiling functions...")
    # Prepare Theano variables for inputs and targets
    real_imgs = T.tensor4('real_imgs')
    fake_imgs = T.tensor4('fake_imgs')
    # Create neural network model
    discriminator = models_uncond.build_discriminator_128(ndf=ndf)
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator, real_imgs)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator, fake_imgs)
    # Create loss expressions
    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1)
            + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()
    
    # Gradients penalty norm 
    if GP_norm is True: 
        alpha = t_rng.uniform((batchSize,1,1,1), low=0.,high=1.)  
        differences = fake_imgs - real_imgs  
        interpolates = real_imgs + (alpha*differences)
        gradients = theano.grad(lasagne.layers.get_output(discriminator, interpolates).sum(), wrt=interpolates) 
        slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1,2,3)))
        gradient_penalty = T.mean((slopes-1.)**2)
 
        D_loss = discriminator_loss +  LAMBDA*gradient_penalty 
        b1_d = 0. 
    else:
        D_loss = discriminator_loss 
        b1_d = b1 
 
    # Create update expressions for training
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    lrtd = theano.shared(lasagne.utils.floatX(lrd))
    updates_d = lasagne.updates.adam(
            D_loss, discriminator_params, learning_rate=lrtd, beta1=b1_d)
    lrt = theano.shared(lasagne.utils.floatX(lr))
    
    # Diversity fitnees 
    Fd = theano.gradient.grad(discriminator_loss, discriminator_params) 
    Fd_score  = beta*T.log(sum(T.sum(T.sqr(x)) for x in Fd))
    
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_d = theano.function([real_imgs, fake_imgs],
                               discriminator_loss,
                               updates=updates_d)
    
    # Compile another function generating some data
    disft_fn = theano.function([real_imgs,fake_imgs],
                               [(real_out).mean(),
                                (fake_out).mean(),
                                Fd_score])
 
    # Finally, launch the training loop.
    print("Starting training...") 
    desc = task + '_' + name 
    print desc
    
    if not os.path.isdir('logs'):
        os.mkdir(os.path.join('logs'))
    f_log = open('logs/%s.ndjson'%desc, 'wb')
    if not os.path.isdir('samples'):
        os.mkdir(os.path.join('samples/'))
    if not os.path.isdir('samples/'+desc):
        os.mkdir(os.path.join('samples/',desc))
    if not os.path.isdir('models'):
        os.mkdir(os.path.join('models/'))
    if not os.path.isdir('models/'+desc):
        os.mkdir(os.path.join('models/',desc))
     
    gen_new_params = [] 
    n_updates = 0
     
    # We iterate over epochs:
    for epoch in range(niter):
        t = time()
	if shuffle_ is True:
	    ids = shuffle(ids) 
	for index_ in iter_data(ids, size=batchSize*kD):
            index = sorted(index_) 
            xmb = trX[index,:,:,:]
	    xmb = Batch(xmb,fineSize,input_nc,flip=flip) 
	    xmb = processing_img(xmb, center=True, scale=True, convert=False)
                        
            rand_idx = random.randint(start,stop-ntf-1) 
            rand_ids = ids[rand_idx:rand_idx+ntf] 
            rand_ids = sorted(rand_ids) 
            sample_xmb = trX[rand_ids,:,:,:]
	    sample_xmb = Batch(sample_xmb,fineSize,input_nc,flip=flip) 
	    sample_xmb = processing_img(sample_xmb, center=True, scale=True, convert=False)
     
            # initial G cluster 
            if epoch + n_updates == 0:
                for can_i in range(0,ncandi): 
                    train_g, gen_fn, generator = create_G(
                            loss_type=loss_type[can_i%nloss],
                            discriminator=discriminator, lr=lr, b1=b1, ngf=ngf) 
                    for _ in range(0,kG):
                        zmb = floatX(np_rng.uniform(-1., 1., size=(batchSize, nz)))
                        cost = train_g(zmb)
                    sample_zmb = floatX(np_rng.uniform(-1., 1., size=(ntf, nz)))
                    gen_imgs = gen_fn(sample_zmb)
 
                    gen_new_params.append(lasagne.layers.get_all_param_values(generator)) 
                    
                    if can_i == 0: 
                        g_imgs_old=gen_imgs
                        fmb = gen_imgs[0:batchSize/ncandi*kD,:,:,:]
                    else: 
                        g_imgs_old = np.append(g_imgs_old,gen_imgs,axis=0)
                        fmb = np.append(fmb,gen_imgs[0:batchSize/ncandi*kD,:,:,:],axis=0)
                #print gen_new_params
                # MODEL G
                noise = T.matrix('noise')
                generator = models_uncond.build_generator_128(noise,ngf=ngf)
                Tgimgs = lasagne.layers.get_output(generator)
                Tfake_out = lasagne.layers.get_output(discriminator, Tgimgs)

                g_loss_logD = lasagne.objectives.binary_crossentropy(Tfake_out, 1).mean()
                g_loss_minimax = -lasagne.objectives.binary_crossentropy(Tfake_out, 0).mean()
                g_loss_ls = T.mean(T.sqr((Tfake_out - 1)))

                g_params = lasagne.layers.get_all_params(generator, trainable=True)

                up_g_logD = lasagne.updates.adam(g_loss_logD, g_params, learning_rate=lrt, beta1=b1)
                up_g_minimax = lasagne.updates.adam(g_loss_minimax, g_params, learning_rate=lrt, beta1=b1)
                up_g_ls = lasagne.updates.adam(g_loss_ls, g_params, learning_rate=lrt, beta1=b1)

                train_g = theano.function([noise],g_loss_logD,updates=up_g_logD)
                train_g_minimax = theano.function([noise],g_loss_minimax,updates=up_g_minimax)
                train_g_ls = theano.function([noise],g_loss_ls,updates=up_g_ls)

                gen_fn = theano.function([noise], lasagne.layers.get_output(
                                        generator,deterministic=True))
            else:
                gen_old_params = gen_new_params
                for can_i in range(0,ncandi):
                    for type_i in range(0,nloss):
                        
                        lasagne.layers.set_all_param_values(generator, gen_old_params[can_i])
                        if loss_type[type_i] == 'trickLogD':
                            for _ in range(0,kG):
                                zmb = floatX(np_rng.uniform(-1., 1., size=(batchSize, nz)))
                                cost = train_g(zmb)
                        elif loss_type[type_i] == 'minimax': 
                            for _ in range(0,kG):
                                zmb = floatX(np_rng.uniform(-1., 1., size=(batchSize, nz)))
                                cost = train_g_minimax(zmb)
                        elif loss_type[type_i] == 'ls': 
                            for _ in range(0,kG):
                                zmb = floatX(np_rng.uniform(-1., 1., size=(batchSize, nz)))
                                cost = train_g_ls(zmb)

                        sample_zmb = floatX(np_rng.uniform(-1., 1., size=(ntf, nz)))
                        gen_imgs = gen_fn(sample_zmb)

                        _, fr_score, fd_score = disft_fn(sample_xmb,gen_imgs)
                        fit = fr_score - fd_score 
                     
                        if can_i*nloss + type_i < ncandi: 
                            idx = can_i*nloss + type_i
                            gen_new_params[idx]=lasagne.layers.get_all_param_values(generator)
                            fitness[idx]=fit
                            fake_rate[idx]=fr_score
                            g_imgs_old[idx*ntf:(idx+1)*ntf,:,:,:]=gen_imgs
                            fmb[idx*batchSize/ncandi*kD:(idx+1)*batchSize/ncandi*kD,:,:,:] = \
                                gen_imgs[0:batchSize/ncandi*kD,:,:,:]
                        else: 
                            fit_com = fitness - fit
                            if min(fit_com) < 0:
                                ids_replace = np.where(fit_com==min(fit_com)) 
                                idr = ids_replace[0][0]
                                fitness[idr]=fit
                                fake_rate[idr]=fr_score

                                gen_new_params[idr] = lasagne.layers.get_all_param_values(generator)

                                g_imgs_old[idr*ntf:(idr+1)*ntf,:,:,:]=gen_imgs
                                fmb[idr*batchSize/ncandi*kD:(idr+1)*batchSize/ncandi*kD,:,:,:] = \
                                    gen_imgs[0:batchSize/ncandi*kD,:,:,:]

                print fake_rate, fitness
                f_log.write(str(fake_rate) + ' '+str(fd_score) +' ' + str(fitness)+ '\n')

            # train D 
	    for xreal,xfake in iter_data(xmb, shuffle(fmb), size=batchSize):
                cost = train_d(xreal, xfake)
            for i in range(0, ncandi):
                xfake = g_imgs_old[i*ntf:(i+1)*ntf,:,:,:]
                xreal = sample_xmb[0:ntf,:,:,:]
                tr, fr, fd = disft_fn(xreal,xfake) 
                if i == 0:
                    fake_rate = np.array([fr])   
                    fitness = np.array([0.])   
                    real_rate = np.array([tr])     
                    FDL = np.array([fd])     
                else:
                    fake_rate = np.append(fake_rate,fr)
                    fitness = np.append(fitness,[0.])
                    real_rate = np.append(real_rate,tr)
                    FDL = np.append(FDL,fd)
            print fake_rate, FDL
            print (n_updates, epoch,real_rate.mean()) 
            n_updates += 1
            f_log.write(str(fake_rate)+' '+str(FDL)+ '\n'+ str(epoch)+' '+str(n_updates)+' '+str(real_rate.mean())+'\n')
            f_log.flush()

	    if n_updates%show_freq == 0:
	        blank_image = Image.new("RGB",(fineSize*8+9,fineSize*8+9))
	        for i in range(8):
                    for ii in range(8):
                        img = g_imgs_old[i*8+ii,:,:,:]
                        img = ImgRescale(img, center=True, scale=True, convert_back=True)
                        blank_image.paste(Image.fromarray(img),(ii*fineSize+ii+1,i*fineSize+i+1)) 
                blank_image.save('samples/%s/%s_%d.png'%(desc,desc,n_updates/save_freq))

            if n_updates%save_freq == 0 and epoch > begin_save - 1:
            # Optionally, you could now dump the network weights to a file like this:
                np.savez('models/%s/gen_%d.npz'%(desc,n_updates/save_freq), *lasagne.layers.get_all_param_values(generator))
                np.savez('models/%s/dis_%d.npz'%(desc,n_updates/save_freq), *lasagne.layers.get_all_param_values(discriminator))


if __name__ == '__main__':
    #if ('--help' in sys.argv) or ('-h' in sys.argv):
    #    print("Trains a DCGAN on MNIST using Lasagne.")
    #    print("Usage: %s [EPOCHS]" % sys.argv[0])
    #    print()
    #    print("EPOCHS: number of training epochs to perform (default: 100)")
    #else:
    #    kwargs = {}
    #    if len(sys.argv) > 1:
    #        kwargs['num_epochs'] = int(sys.argv[1])
    main()
