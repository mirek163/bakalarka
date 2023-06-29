from __future__ import print_function, division

import os
#mport cv2
#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import ELU, PReLU, LeakyReLU
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt


import glob
from PIL import Image

import noise as ns
from noise import snoise2
import random

import numpy as np

class GAN():
    def __init__(self, noise_type='random'):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.noise_type = noise_type

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(noise_type=self.noise_type)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self, noise_type):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))  # Set training=True during training
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))  # Set training=True during training
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))  # Set training=True during training


        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        if noise_type == 'perlin':
            perlin_noise = np.empty((self.latent_dim,))
            for i in range(self.latent_dim):
                perlin_noise[i] = ns.pnoise2(i, 0.1)
            noise = perlin_noise
            noise = noise.reshape((1, -1))  # Reshape the noise tensor to have a known shape

        elif noise_type == 'simplex':
            simplex_noise = np.empty((self.latent_dim,))
            for i in range(self.latent_dim):
                simplex_noise[i] = ns.snoise2(i, 0.1)
            noise = simplex_noise
            noise = noise.reshape((1, -1))  # Reshape the noise tensor to have a known shape

        else:
            noise = noise

        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50, noise_type='random', mode='generate'):

        image_files = glob.glob("C:/.develop/bakalarka/data/crack/*.jpg")
        X_train = []
        for image_file in image_files:
            image = Image.open(image_file).convert('RGB')  # převed na grayscale

            # Uprav rozměry
            width, height = image.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            image = image.crop((left, top, right, bottom))

            # Resize the image to 64x64
            image = image.resize((64, 64))

            image = np.array(image)
            X_train.append(image)

        X_train = np.array(X_train)
        X_train = X_train / 127.5 - 1.
        X_train = X_train.reshape((-1, self.img_rows, self.img_cols, self.channels))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            myNoise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(myNoise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            myNoise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            for _ in range(3):
                g_loss = self.combined.train_on_batch(myNoise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, noise_type)

    def sample_images(self, epoch, noise_type='random'):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        if noise_type == 'perlin':
            perlin_noise = np.empty((r * c, self.latent_dim))
            for i in range(r * c):
                for j in range(self.latent_dim):
                    perlin_noise[i, j] = ns.pnoise2(i, j)
            noise = perlin_noise

        elif noise_type == 'simplex':
            simplex_noise = np.empty((r * c, self.latent_dim))
            for i in range(r * c):
                for j in range(self.latent_dim):
                    simplex_noise[i, j] = ns.snoise2(i, j)
            noise = simplex_noise

        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i,j].axis('off')
                cnt += 1

        output_dir = "C:/.develop/bakalarka/data/output/" + noise_type
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

        fig.savefig(os.path.join(output_dir, "%d.png" % epoch))
        plt.close()

        # Ulož si své váhy
        weights_dir = 'C:/.develop/bakalarka\data/weights/'
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, 'weights.h5')
        gan.generator.save_weights(weights_path)


if __name__ == '__main__':
    gan = GAN()

    # trénovací / generovací (train/generate)
    mode = 'train'
    # random,perlin noise, simplex noise (random/perlin/simplex)
    noise_type = 'simplex'



    if mode == 'train':
        gan.train(epochs=300000, batch_size=32, sample_interval=200, noise_type=noise_type)

    elif mode == 'generate':
        # načti váhy
        weights_path = 'data/weights/weights.h5'
        gan.build_generator()
        gan.generator.load_weights(weights_path)

        # Generuj
        epoch_number = 10000  # Počet epoch
        noise_type = noise_type  # Typ noise
        gan.sample_images(epoch_number, noise_type)