from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, DropoutLayer, Deconv2DLayer, BatchNormLayer, NonlinearityLayer, ElemwiseSumLayer, ConcatLayer, FlattenLayer, Pool2DLayer, Upscale2DLayer
from lasagne.nonlinearities import sigmoid, LeakyRectify, sigmoid, tanh, softmax, elu
from lasagne.nonlinearities import rectify as relu
from lasagne.layers import Conv2DLayer
from theano.tensor import TensorType
from lasagne.init import Normal, HeNormal
import theano
from theano import tensor as T

def build_generator_toy(noise=None, nd=512):
    InputNoise = InputLayer(shape=(None, 2), input_var=noise)
    print ("Gen input:", InputNoise.output_shape)
    gnet0 = DenseLayer(InputNoise, nd, W=Normal(0.02), nonlinearity=relu)
    print ("Gen fc0:", gnet0.output_shape)
    gnet1 = DenseLayer(gnet0, nd, W=Normal(0.02), nonlinearity=relu)
    print ("Gen fc1:", gnet1.output_shape)
    gnet2 = DenseLayer(gnet1, nd, W=Normal(0.02), nonlinearity=relu)
    print ("Gen fc2:", gnet2.output_shape)
    gnetout = DenseLayer(gnet2, 2, W=Normal(0.02), nonlinearity=None)
    print ("Gen output:", gnetout.output_shape)
    return gnetout 

def build_discriminator_toy(image=None, nd=512, GP_norm=None):
    Input = InputLayer(shape=(None, 2), input_var=image)
    print ("Dis input:", Input.output_shape)
    dis0 = DenseLayer(Input, nd, W=Normal(0.02), nonlinearity=relu)
    print ("Dis fc0:", dis0.output_shape)
    if GP_norm is True:
        dis1 = DenseLayer(dis0, nd, W=Normal(0.02), nonlinearity=relu)
    else:
        dis1 = batch_norm(DenseLayer(dis0, nd, W=Normal(0.02), nonlinearity=relu))
    print ("Dis fc1:", dis1.output_shape)
    if GP_norm is True:
        dis2 = batch_norm(DenseLayer(dis1, nd, W=Normal(0.02), nonlinearity=relu))
    else:
        dis2 = DenseLayer(dis1, nd, W=Normal(0.02), nonlinearity=relu)
    print ("Dis fc2:", dis2.output_shape)
    disout = DenseLayer(dis2, 1, W=Normal(0.02), nonlinearity=sigmoid)
    print ("Dis output:", disout.output_shape)
    return disout 

def build_generator_32(noise=None, ngf=128):
    # noise input 
    InputNoise = InputLayer(shape=(None, 100), input_var=noise)
    #FC Layer 
    gnet0 = DenseLayer(InputNoise, ngf*4*4*4, W=Normal(0.02), nonlinearity=relu)
    print ("Gen fc1:", gnet0.output_shape)
    #Reshape Layer
    gnet1 = ReshapeLayer(gnet0,([0],ngf*4,4,4))
    print ("Gen rs1:", gnet1.output_shape)
    # DeConv Layer
    gnet2 = Deconv2DLayer(gnet1, ngf*2, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=relu)
    print ("Gen deconv1:", gnet2.output_shape)
    # DeConv Layer
    gnet3 = Deconv2DLayer(gnet2, ngf, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=relu)
    print ("Gen deconv2:", gnet3.output_shape)
    # DeConv Layer
    gnet4 = Deconv2DLayer(gnet3, 3, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=tanh)
    print ("Gen output:", gnet4.output_shape)
    return gnet4 

def build_discriminator_32(image=None,ndf=128):
    lrelu = LeakyRectify(0.2)
    # input: images
    InputImg = InputLayer(shape=(None, 3, 32, 32), input_var=image)
    print ("Dis Img_input:", InputImg.output_shape)
    # Conv Layer
    dis1 = Conv2DLayer(InputImg, ndf, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu)
    print ("Dis conv1:", dis1.output_shape)
    # Conv Layer
    dis2 = batch_norm(Conv2DLayer(dis1, ndf*2, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv2:", dis2.output_shape)
    # Conv Layer
    dis3 = batch_norm(Conv2DLayer(dis2, ndf*4, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv3:", dis3.output_shape)
    # Conv Layer
    dis4 = DenseLayer(dis3, 1, W=Normal(0.02), nonlinearity=sigmoid)
    print ("Dis output:", dis4.output_shape)
    return dis4 

def build_generator_64(noise=None, ngf=128):
    # noise input 
    InputNoise = InputLayer(shape=(None, 100), input_var=noise)
    #FC Layer 
    gnet0 = DenseLayer(InputNoise, ngf*8*4*4, W=Normal(0.02), nonlinearity=relu)
    print ("Gen fc1:", gnet0.output_shape)
    #Reshape Layer
    gnet1 = ReshapeLayer(gnet0,([0],ngf*8,4,4))
    print ("Gen rs1:", gnet1.output_shape)
    # DeConv Layer
    gnet2 = Deconv2DLayer(gnet1, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=relu)
    print ("Gen deconv2:", gnet2.output_shape)
    # DeConv Layer
    gnet3 = Deconv2DLayer(gnet2, ngf*4, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=relu)
    print ("Gen deconv3:", gnet3.output_shape)
    # DeConv Layer
    gnet4 = Deconv2DLayer(gnet3, ngf*4, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=relu)
    print ("Gen deconv4:", gnet4.output_shape)
    # DeConv Layer
    gnet5 = Deconv2DLayer(gnet4, ngf*2, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=relu)
    print ("Gen deconv5:", gnet5.output_shape)
    # DeConv Layer
    gnet6 = Deconv2DLayer(gnet5, 3, (3,3), (1,1), crop='same', W=Normal(0.02),nonlinearity=tanh)
    print ("Gen output:", gnet6.output_shape)
    return gnet6 

def build_discriminator_64(image=None,ndf=128):
    lrelu = LeakyRectify(0.2)
    # input: images
    InputImg = InputLayer(shape=(None, 3, 64, 64), input_var=image)
    print ("Dis Img_input:", InputImg.output_shape)
    # Conv Layer
    dis1 = Conv2DLayer(InputImg, ndf, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu)
    print ("Dis conv1:", dis1.output_shape)
    # Conv Layer
    dis2 = batch_norm(Conv2DLayer(dis1, ndf*2, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv2:", dis2.output_shape)
    # Conv Layer
    dis3 = batch_norm(Conv2DLayer(dis2, ndf*4, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv3:", dis3.output_shape)
    # Conv Layer
    dis4 = batch_norm(Conv2DLayer(dis3, ndf*8, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv3:", dis4.output_shape)
    # Conv Layer
    dis5 = DenseLayer(dis4, 1, W=Normal(0.02), nonlinearity=sigmoid)
    print ("Dis output:", dis5.output_shape)
    return dis5 

def build_generator_128(noise=None, ngf=128):
    lrelu = LeakyRectify(0.2)
    # noise input 
    InputNoise = InputLayer(shape=(None, 100), input_var=noise)
    #FC Layer 
    gnet0 = DenseLayer(InputNoise, ngf*16*4*4, W=Normal(0.02), nonlinearity=lrelu)
    print ("Gen fc1:", gnet0.output_shape)
    #Reshape Layer
    gnet1 = ReshapeLayer(gnet0,([0],ngf*16,4,4))
    print ("Gen rs1:", gnet1.output_shape)
    # DeConv Layer
    gnet2 = Deconv2DLayer(gnet1, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=lrelu)
    print ("Gen deconv1:", gnet2.output_shape)
    # DeConv Layer
    gnet3 = Deconv2DLayer(gnet2, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=lrelu)
    print ("Gen deconv2:", gnet3.output_shape)
    # DeConv Layer
    gnet4 = Deconv2DLayer(gnet3, ngf*4, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=lrelu)
    print ("Gen deconv3:", gnet4.output_shape)
    # DeConv Layer
    gnet5 = Deconv2DLayer(gnet4, ngf*4, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=lrelu)
    print ("Gen deconv4:", gnet5.output_shape)
    # DeConv Layer
    gnet6 = Deconv2DLayer(gnet5, ngf*2, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=lrelu)
    print ("Gen deconv5:", gnet6.output_shape)
    # DeConv Layer
    gnet7 = Deconv2DLayer(gnet6, 3, (3,3), (1,1), crop='same', W=Normal(0.02),nonlinearity=tanh)
    print ("Gen output:", gnet7.output_shape)
    return gnet7 

def build_discriminator_128(image=None,ndf=128):
    lrelu = LeakyRectify(0.2)
    # input: images
    InputImg = InputLayer(shape=(None, 3, 128, 128), input_var=image)
    print ("Dis Img_input:", InputImg.output_shape)
    # Conv Layer
    dis1 = Conv2DLayer(InputImg, ndf, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu)
    print ("Dis conv1:", dis1.output_shape)
    # Conv Layer
    dis2 = batch_norm(Conv2DLayer(dis1, ndf*2, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv2:", dis2.output_shape)
    # Conv Layer
    dis3 = batch_norm(Conv2DLayer(dis2, ndf*4, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv3:", dis3.output_shape)
    # Conv Layer
    dis4 = batch_norm(Conv2DLayer(dis3, ndf*8, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv3:", dis4.output_shape)
    # Conv Layer
    dis5 = batch_norm(Conv2DLayer(dis4, ndf*16, (4,4), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Dis conv4:", dis5.output_shape)
    # Conv Layer
    dis6 = DenseLayer(dis5, 1, W=Normal(0.02), nonlinearity=sigmoid)
    print ("Dis output:", dis6.output_shape)
    return dis6 

