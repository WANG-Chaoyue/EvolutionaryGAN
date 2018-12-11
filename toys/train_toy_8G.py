import sys
sys.path.append('..')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave, imread

import os
import numpy as np
import theano
import theano.tensor as T
import json
import lasagne
import shutil
import h5py 

from time import time
from PIL import Image
from lib.data_utils import processing_img, convert_img_back, convert_img, Batch, shuffle, iter_data, ImgRescale, OneHot
from lib.theano_utils import floatX, sharedX
from lib.rng import py_rng, np_rng, t_rng
from models import models_uncond
from dataset.datasets import toy_dataset 

def create_G(loss_type=None, discriminator=None, lr=0.0002, b1=0.5, DIM=64): 
    noise = T.matrix('noise')
    generator = models_uncond.build_generator_toy(noise,nd=DIM)
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

def generate_image(true_dist, generate_dist, num=0, desc=None):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    #plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    #if not FIXED_GENERATOR:
    plt.scatter(generate_dist[:, 0], generate_dist[:, 1], c='green', marker='+')

    if not os.path.isdir('tmp'):
        os.mkdir(os.path.join('tmp/'))
    if not os.path.isdir('tmp/'+desc):
        os.mkdir(os.path.join('tmp/',desc))

    #plt.savefig('tmp/' + DATASET + '/' + 'frame' + str(frame_index[0]) + '.jpg')
    plt.savefig('tmp/' + desc + '/frame_'+ str(num) + '.jpg')

    #frame_index[0] += 1

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main():
    # Parameters
    task = 'toy'
    name = '8G'
    
    DIM=512
    begin_save = 0 
    loss_type = ['trickLogD','minimax','ls'] 
    nloss = 3
    DATASET = '8gaussians'
    batchSize = 64 
   
    ncandi = 1 
    kD = 1             # # of discrim updates for each gen update
    kG = 1            # # of discrim updates for each gen update
    ntf = 256 
    b1 = 0.5          # momentum term of adam
    nz = 2          # # of dim for Z
    niter = 4       # # of iter at starting learning rate
    lr = 0.0001       # initial learning rate for adam G
    lrd = 0.0001       # initial learning rate for adam D
    N_up = 100000 
    save_freq = 10000 
    show_freq = 10000 
    test_deterministic = True   
    beta = 1.
    GP_norm = False     # if use gradients penalty on discriminator
    LAMBDA = 2.       # hyperparameter of GP 
   
    # Load the dataset
   
    # MODEL D
    print("Building model and compiling functions...")
    # Prepare Theano variables for inputs and targets
    real_imgs = T.matrix('real_imgs')
    fake_imgs = T.matrix('fake_imgs')
    # Create neural network model
    discriminator = models_uncond.build_discriminator_toy(nd=DIM, GP_norm=GP_norm)
    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator, real_imgs)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator, fake_imgs)
    # Create loss expressions
    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1)
            + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()
 
    # Gradients penalty norm 
    if GP_norm is True: 
        alpha = t_rng.uniform((batchSize,1), low=0.,high=1.)  
        differences = fake_imgs - real_imgs  
        interpolates = real_imgs + (alpha*differences)
        gradients = theano.grad(lasagne.layers.get_output(discriminator, interpolates).sum(), wrt=interpolates) 
        slopes = T.sqrt(T.sum(T.sqr(gradients), axis=(1)))
        gradient_penalty = T.mean((slopes-1.)**2)
 
        D_loss = discriminator_loss +  LAMBDA*gradient_penalty 
        b1_d = 0. 
    else:
        D_loss = discriminator_loss 
        b1_d = 0. 
 
    # Create update expressions for training
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    lrtd = theano.shared(lasagne.utils.floatX(lrd))
    updates_d = lasagne.updates.adam(
            D_loss, discriminator_params, learning_rate=lrtd, beta1=b1_d)
    lrt = theano.shared(lasagne.utils.floatX(lr))

    # Fd Socre 
    Fd = theano.gradient.grad(discriminator_loss, discriminator_params) 
    Fd_score  = beta*T.log(sum(T.sum(T.sqr(x)) for x in Fd))
     
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_d = theano.function([real_imgs, fake_imgs],
                               discriminator_loss,
                               updates=updates_d)
    # Compile another function generating some data
    dis_fn = theano.function([real_imgs,fake_imgs],[(fake_out).mean(),Fd_score])
    disft_fn = theano.function([real_imgs,fake_imgs],
                               [real_out.mean(),
                                fake_out.mean(),
                                (real_out>0.5).mean(),
                                (fake_out>0.5).mean(),
                                Fd_score])
 
    # Finally, launch the training loop.
    print("Starting training...") 
    desc = task + '_' + name 
    print desc
    
    if not os.path.isdir('logs'):
        os.mkdir(os.path.join('logs'))
    f_log = open('logs/%s.ndjson'%desc, 'wb')
    if not os.path.isdir('models'):
        os.mkdir(os.path.join('models/'))
    if not os.path.isdir('models/'+desc):
        os.mkdir(os.path.join('models/',desc))
     
    gen_new_params = [] 
     
    # We iterate over epochs:
    for n_updates in range(N_up):
        xmb = toy_dataset(DATASET=DATASET, size=batchSize*kD) 
        xmb = xmb[0:batchSize*kD] 
        # initial G cluster 
        if n_updates == 0:
            for can_i in range(0,ncandi): 
                train_g, gen_fn, generator = create_G(
                        loss_type=loss_type[can_i%nloss],
                        discriminator=discriminator, lr=lr, b1=b1, DIM=DIM) 
                for _ in range(0,kG):
                    zmb = floatX(np_rng.uniform(-1., 1., size=(batchSize, nz)))
                    cost = train_g(zmb)
                sample_zmb = floatX(np_rng.uniform(-1., 1., size=(ntf, nz)))
                gen_imgs = gen_fn(sample_zmb)
 
                gen_new_params.append(lasagne.layers.get_all_param_values(generator)) 
                    
                if can_i == 0: 
                    g_imgs_old=gen_imgs
                    fmb = gen_imgs[0:batchSize/ncandi*kD,:]
                else: 
                    g_imgs_old = np.append(g_imgs_old,gen_imgs,axis=0)
                    fmb = np.append(fmb,gen_imgs[0:batchSize/ncandi*kD,:],axis=0)
            #print gen_new_params
            # MODEL G
            noise = T.matrix('noise')
            generator = models_uncond.build_generator_toy(noise,nd=DIM)
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
                    frr_score, fd_score = dis_fn(xmb[0:ntf],gen_imgs)
                    #frr = frr[0]
                    frr = frr_score - fd_score 
                    if can_i*nloss + type_i < ncandi: 
                        idx = can_i*nloss + type_i
                        gen_new_params[idx]=lasagne.layers.get_all_param_values(generator)
                        fake_rate[idx]=frr
                        g_imgs_old[idx*ntf:(idx+1)*ntf,:]=gen_imgs
                        fmb[idx*batchSize/ncandi*kD:(idx+1)*batchSize/ncandi*kD,:] = \
                            gen_imgs[0:batchSize/ncandi*kD,:]
                    else: 
                        fr_com = fake_rate - frr
                        if min(fr_com) < 0:
                            ids_replace = np.where(fr_com==min(fr_com)) 
                            idr = ids_replace[0][0]
                            fake_rate[idr]=frr

                            gen_new_params[idr] = lasagne.layers.get_all_param_values(generator)

                            g_imgs_old[idr*ntf:(idr+1)*ntf,:]=gen_imgs
                            fmb[idr*batchSize/ncandi*kD:(idr+1)*batchSize/ncandi*kD,:] = \
                                gen_imgs[0:batchSize/ncandi*kD,:]

        sample_xmb = toy_dataset(DATASET=DATASET, size=ncandi*ntf) 
        sample_xmb = sample_xmb[0:ncandi*ntf] 
        for i in range(0, ncandi):
            xfake = g_imgs_old[i*ntf:(i+1)*ntf,:]
            xreal = sample_xmb[i*ntf:(i+1)*ntf,:]
            tr, fr, trp, frp, fdscore = disft_fn(xreal,xfake) 
            if i == 0:
                fake_rate = np.array([fr])   
                real_rate = np.array([tr])     
                fake_rate_p = np.array([frp])   
                real_rate_p = np.array([trp])     
                FDL = np.array([fdscore])     
            else:
                fake_rate = np.append(fake_rate,fr)
                real_rate = np.append(real_rate,tr)
                fake_rate_p = np.append(fake_rate_p,frp)
                real_rate_p = np.append(real_rate_p,trp)
                FDL = np.append(FDL,fdscore)
        print fake_rate, fake_rate_p, FDL
        print (n_updates, real_rate.mean(), real_rate_p.mean()) 
        f_log.write(str(fake_rate)+' '+str(fake_rate_p)+'\n'+ str(n_updates) + ' ' + str(real_rate.mean())+ ' ' +str(real_rate_p.mean())+'\n')
        f_log.flush()

        # train D 
	for xreal,xfake in iter_data(xmb, shuffle(fmb), size=batchSize):
            cost = train_d(xreal, xfake)

	if n_updates%show_freq == 0:
            s_zmb = floatX(np_rng.uniform(-1., 1., size=(512, nz)))
            g_imgs = gen_fn(s_zmb)
            xmb = toy_dataset(DATASET=DATASET, size=512) 
            generate_image(xmb,g_imgs,n_updates/save_freq,desc)

        if n_updates%save_freq == 0 and n_updates > begin_save - 1:
        # Optionally, you could now dump the network weights to a file like this:
        #    np.savez('models/%s/gen_%d.npz'%(desc,n_updates/save_freq), *lasagne.layers.get_all_param_values(generator))
        #    np.savez('models/%s/dis_%d.npz'%(desc,n_updates/save_freq), *lasagne.layers.get_all_param_values(discriminator))

if __name__ == '__main__':
    '''
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
    '''
    main()
