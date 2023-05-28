'''
Cycle GAN
Based on the code by Jason Brownlee from his blogs on https://machinelearningmastery.com/

custom_argument can be found at:
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
'''

from random import random
import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model,Input
from keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,Activation,Concatenate
from custom_argument import InstanceNormalization  
import matplotlib.pyplot as plt
import glob
from skimage import io
from PIL import Image
import cv2
from keras.models import load_model

def Discriminator(image_shape):

	init = RandomNormal(stddev=0.02)
    
	img = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(img)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	output = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)

	model = Model(img,output)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

def residual_block(num, input_layer):

	init = RandomNormal(stddev=0.02)

	res = Conv2D(num, (3,3), padding='same', kernel_initializer=init)(input_layer)
	res = InstanceNormalization(axis=-1)(res)
	res = Activation('relu')(res)

	res = Conv2D(num, (3,3), padding='same', kernel_initializer=init)(res)
	res = InstanceNormalization(axis=-1)(res)
	res = Concatenate()([res, input_layer])
	return res

# Network: c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3

def Generator(image_shape, num_resnet=9):

	init = RandomNormal(stddev=0.02)

	img = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(img)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(num_resnet):
		g = residual_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out = Activation('tanh')(g)

	model = Model(img, out)
	return model

def cycle_GANs(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True
	# discriminator and second generator as non-trainable
    g_model_2.trainable = False
    d_model.trainable = False
	
	# adversarial loss
    adv_input = Input(shape=image_shape)
    G1_output = g_model_1(adv_input)
    D_output = d_model(G1_output)
	# identity loss
    id_input = Input(shape=image_shape)
    id_output = g_model_1(id_input)
	# cycle loss - forward
    output_f = g_model_2(G1_output)
	# cycle loss - backward
    G2_output = g_model_2(id_input)
    output_b = g_model_1(G2_output)
    
    model = Model([adv_input, id_input], [D_output, id_output, output_f, output_b])
    
    opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'],loss_weights=[1, 5, 10, 10], optimizer=opt)
    
    return model

def observe_performance(step, g_model,image,name):

    ix = np.random.randint(0, image.shape[0], 1)
    I=image[ix]
    G_fake=g_model.predict(I)
	# scale all pixels from [-1,1] to [0,1]
    i = (I + 1) / 2.0
    gf = (G_fake + 1) / 2.0
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(i[0,:,:,:])
    plt.subplot(1,2,2)
    plt.imshow(gf[0,:,:,:])
    
    figure_name = './eval/%s_generated_plot_%06d.png' % (name, (step+1))
    plt.savefig(figure_name)
    plt.close()
    

def cache_images(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = np.random.randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return np.asarray(selected)

# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, image1,image2, epochs=60):
    
    n_patch = d_model_A.output_shape[1]
    poolA, poolB = list(), list() # prepare image pool for fake images
    n_steps = len(image1) * epochs
    for i in range(n_steps):
        
        ix = np.random.randint(0, image1.shape[0], 1) # select a batch of real samples from each domain (A and B)
        x_real_A=image1[ix]
        y_real_A = np.ones((1, n_patch, n_patch, 1))
        
        iy = np.random.randint(0, image2.shape[0], 1)
        x_real_B=image2[iy]
        y_real_B = np.ones((1, n_patch, n_patch, 1))
        
        x_fake_A=g_model_BtoA.predict(x_real_B)
        y_fake_A=np.zeros((1, n_patch, n_patch, 1))
        
        x_fake_B=g_model_AtoB.predict(x_real_A)
        y_fake_B=np.zeros((1, n_patch, n_patch, 1))
        
		# update fake images in the pool. Remember that the paper suggstes a buffer of 50 images
        x_fake_A = cache_images(poolA, x_fake_A)
        x_fake_B = cache_images(poolB, x_fake_B)
        
		# update generator B->A via the composite model
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([x_real_B, x_real_A], [y_real_A, x_real_A, x_real_B, x_real_A])
		# update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(x_real_A, y_real_A)
        dA_loss2 = d_model_A.train_on_batch(x_fake_A, y_fake_A)
		
        # update generator A->B via the composite model
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([x_real_A, x_real_B], [y_real_B, x_real_B, x_real_A, x_real_B])
		# update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(x_real_B, y_real_B)
        dB_loss2 = d_model_B.train_on_batch(x_fake_B, y_fake_B)
        # summarize performance
        #for x images: 1 epoch would be x iterations
        print('Iteration>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
        
        if (i+1) % (len(image1) * 1) == 0: #If batch size (total images)=100, performance will be summarized after every 75th iteration.
			# plot A->B translation
            observe_performance(i, g_model_AtoB, image1, 'AtoB')
			# plot B->A translation
            observe_performance(i, g_model_BtoA, image2, 'BtoA')
            
        if (i+1) % (len(image1) * 5) == 0:
            filename1 = './weights/g_model_AtoB_%06d.h5' % (i+1) # save the models
            g_model_AtoB.save(filename1)
            
            filename2 = './weights/g_model_BtoA_%06d.h5' % (i+1)
            g_model_BtoA.save(filename2)
            print('>Saved: %s and %s' % (filename1, filename2))
            
            
img1_path=glob.glob('./glass/*')
img2_path=glob.glob('./no_glass/*')

I1,I2=[],[]
for i in range(1500):
    img1=io.imread(img1_path[i])
    img1=(img1-127.5)/127.5
    img2=io.imread(img2_path[i])
    img2=(img2-127.5)/127.5
    I1.append(img1)
    I2.append(img2)
    
I1=np.array(I1)
I2=np.array(I2)

image_shape = I1.shape[1:]
# generator: A -> B
g_model_AtoB = Generator(image_shape)
# generator: B -> A
g_model_BtoA = Generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = Discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = Discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = cycle_GANs(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = cycle_GANs(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

from datetime import datetime 
start1 = datetime.now() 
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, I1,I2, epochs=60)
stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

