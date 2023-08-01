from __future__ import print_function, division  # Pro kompatibilitu s Python 2
import os  # Manipulace se souborovým systémem

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import traceback  # Výpis trasování chyb
from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose  # Vrstvy modelu Keras
from keras.layers import BatchNormalization, LeakyReLU, Conv2D
from keras.layers import Dropout
from keras.models import Sequential, Model  # Modely Keras
from keras.optimizers import Adam  # Optimalizátor Adam
import matplotlib.pyplot as plt  # Vykreslování obrázků
import glob  # Vyhledávání souborů v adresáři
from PIL import Image  # Práce s obrázky
import noise as ns  # Generování šumu
import numpy as np  # Matematické operace s poli
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

import math
import random
from itertools import product

# import keras_preprocessing.image.image_data_generator as ImageDataGenerator

DATASET = 'data/input/region/'
WEIGHT_PATH = 'data/weights/'
OUTPUT_PATH = 'data/output/'
WIDTH = 32
LATENT_DIM = WIDTH * WIDTH
DATASET_NUMBER = 20000
BATCH_SIZE = 32
EPOCH_TO_SAVE_WEIGHT = 1000  # Musí být větší než sample interval
OCTAVES, PERSISTENCE, LACUNARITY = 8, 0.5, 1.5
SIZE = 256  # 256x256


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


class GAN():

    def __init__(self, noise_type='random'):
        # Velikost vstupního obrázku
        self.img_rows = SIZE
        self.img_cols = SIZE
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Dimenze latentního vektoru
        self.noise_type = noise_type
        self.DATASET = DATASET
        self.LATENT_DIM = LATENT_DIM

        optimizer = Adam(0.0002, 0.5, 0.9)  # 0.0002, 0.5

        # Sestavení a kompilace diskriminátoru
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Sestavení generátoru
        self.generator = self.build_generator(noise_type=self.noise_type)

        # Generátor přijímá šum jako vstup a generuje obrázky
        if noise_type == "random" or noise_type == "perlin" or noise_type == "simplex":
            z = Input(shape=(self.LATENT_DIM,))
        else:
            z = Input(shape=(WIDTH, WIDTH, 3))
        img = self.generator(z)

        # Při kombinovaném modelu budeme trénovat pouze generátor
        self.discriminator.trainable = False

        # Diskriminátor přijímá vygenerované obrázky jako vstup a určuje jejich validitu
        validity = self.discriminator(img)

        # Kombinovaný model (generátor a diskriminátor)
        # Trénuje generátor, aby přelstil diskriminátor
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self, noise_type):
        model = Sequential()

        # Generator layers
        if noise_type == "random" or noise_type == "perlin" or noise_type == "simplex":
            model.add(Dense(128 * 16 * 16))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Reshape((16, 16, 128)))
            model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.4))

            model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.4))

            model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.4))

            model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

            model.add(Conv2DTranspose(3, kernel_size=5, strides=1, padding='same', activation='tanh'))
        else:
            inputs = tf.keras.layers.Input(shape=[WIDTH, WIDTH, 3])

            down_stack = [
                downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
                downsample(128, 4),  # (batch_size, 64, 64, 128)
                downsample(256, 4),  # (batch_size, 32, 32, 256)
                downsample(512, 4),  # (batch_size, 16, 16, 512)
                downsample(512, 4),  # (batch_size, 8, 8, 512)
                downsample(512, 4),  # (batch_size, 4, 4, 512)
                downsample(512, 4),  # (batch_size, 2, 2, 512)
                downsample(512, 4),  # (batch_size, 1, 1, 512)
            ]

            up_stack = [
                upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
                upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
                upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
                upsample(512, 4),  # (batch_size, 16, 16, 1024)
                upsample(256, 4),  # (batch_size, 32, 32, 512)
                upsample(128, 4),  # (batch_size, 64, 64, 256)
                upsample(64, 4),  # (batch_size, 128, 128, 128)
            ]

            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(3, 4,
                                                   strides=2,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   activation='tanh')  # (batch_size, 256, 256, 3)

            x = inputs

            # Downsampling through the model
            skips = []
            for down in down_stack:
                x = down(x)
                skips.append(x)

            skips = reversed(skips[:-1])

            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                x = tf.keras.layers.Concatenate()([x, skip])
            x = last(x)
            return tf.keras.Model(inputs=inputs, outputs=x)

        noise = Input(shape=(self.LATENT_DIM,))
        img = model(noise)
        model.summary()

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        # Discriminator layers
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(256, 256, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))  # Dropout layer
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))  # Dropout layer
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))  # Dropout layer
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=(256, 256, 3))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=100, sample_interval=50, noise_type='random', epoch=0):
        """
        Metoda pro trénování modelu.

        Parametry:
        - epochs: Celkový počet epoch (iterací) pro trénování modelu.
        - batch_size: Velikost dávky obrázků použitých při každé iteraci trénování.
        - sample_interval: Počet epoch mezi výstupy generátoru (generované obrázky) pro vizuální kontrolu.
        - noise_type: Typ šumu použitý při generování obrázků. Možné hodnoty jsou 'random' (náhodný šum),
          'perlin' (Perlinův šum) a 'simplex' (simplexový šum).
        """

        try:
            # Načtení obrázků ze složky
            image_files = glob.glob(self.DATASET + '*.png')
            X_train = []
            random.shuffle(image_files)
            image_files = image_files[:DATASET_NUMBER]

            for image_file in image_files:
                try:
                    image = Image.open(image_file).convert('RGB')
                    width, height = image.size
                    size = min(width, height)
                    left = (width - size) // 2
                    top = (height - size) // 2
                    right = left + size
                    bottom = top + size
                    image = image.crop((left, top, right, bottom))
                    image = image.resize((256, 256))
                    image = np.array(image)
                    X_train.append(image)
                except Exception as e:
                    print("Při načítání obrázku ze sady došlo k chybě:", image_file)
                    traceback.print_exc()

            X_train = np.array(X_train)
            X_train = X_train / 127.5 - 1.  # Normalizace hodnot obrázků do rozsahu [-1, 1]
            X_train = X_train.reshape((-1, self.img_rows, self.img_cols, self.channels))  # taky není potřeba

            # Definování pravdivostních hodnot pro trénování diskriminátoru
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            for epoch in range(epochs):
                # ---------------------
                #  Trénování diskriminátoru
                # ---------------------

                # Výběr náhodných obrázků z trénovacího datasetu
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Generování šumu pro generátor

                my_noise = np.random.normal(0, 1, (batch_size, self.LATENT_DIM))

                if noise_type == 'perlin':
                    for i in range(batch_size):
                        base = np.random.uniform(0, 1000)
                        for j in range(LATENT_DIM):
                            x = (base + j % width) / 4
                            y = (base + j / width) / 4
                            my_noise[i][j] = ns.pnoise2(x, y, octaves=OCTAVES, persistence=PERSISTENCE,
                                                        lacunarity=LACUNARITY)

                elif noise_type == 'perlin2D':
                    width = 64
                    for i in range(batch_size):
                        base = np.random.uniform(0, 1000)
                        for j in range(LATENT_DIM):
                            x = (base + j % width) / 4
                            y = (base + j / width) / 4
                            my_noise[i][j] = ns.pnoise2(x, y, octaves=OCTAVES, persistence=PERSISTENCE,
                                                        lacunarity=LACUNARITY)
                    my_noise = np.reshape(my_noise, (BATCH_SIZE, WIDTH, WIDTH, 3))

                elif noise_type == 'simplex':
                    for i in range(batch_size):
                        base = np.random.uniform(0, 1000)
                        for j in range(LATENT_DIM):
                            x = (base + j % width) / 4
                            y = (base + j / width) / 4
                            my_noise[i][j] = ns.snoise2(x, y, octaves=OCTAVES, persistence=PERSISTENCE,
                                                        lacunarity=LACUNARITY)

                # Generování obrázků pomocí generátoru
                # gen_imgs = self.generator.predict(my_noise.reshape(batch_size, self.LATENT_DIM))
                gen_imgs = self.generator.predict(my_noise)

                # Úprava tvaru obrázků pro diskriminátor
                imgs = np.reshape(imgs, (batch_size, self.img_rows, self.img_cols, self.channels))

                # Trénování diskriminátoru na reálných a generovaných obrázcích
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

                # Celková ztráta diskriminátoru
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Trénování generátoru
                # ---------------------
                g_loss = 0
                if d_loss[1] > 0.9:
                    rep = 4  # 4
                else:
                    rep = 1
                # rep=1
                for j in range(rep):
                    generator_batch_multiplicator = 1
                    batch_size_generator = batch_size * generator_batch_multiplicator

                    my_noise_2 = np.random.normal(0, 1, (batch_size_generator, self.LATENT_DIM))

                    if noise_type == 'perlin':
                        for i in range(batch_size):
                            base = np.random.uniform(0, 1000)
                            for j in range(LATENT_DIM):
                                x = (base + j % width) / 4
                                y = (base + j / width) / 4
                                my_noise_2[i][j] = ns.pnoise2(x, y, octaves=OCTAVES, persistence=PERSISTENCE,
                                                              lacunarity=LACUNARITY)

                    elif noise_type == 'perlin2D':
                        width = 64
                        for i in range(batch_size):
                            base = np.random.uniform(0, 1000)
                            for j in range(LATENT_DIM):
                                x = (base + j % width) / 4
                                y = (base + j / width) / 4
                                my_noise_2[i][j] = ns.pnoise2(x, y, octaves=OCTAVES, persistence=PERSISTENCE,
                                                              lacunarity=LACUNARITY)
                        my_noise_2 = np.reshape(my_noise, (BATCH_SIZE, WIDTH, WIDTH, 3))


                    elif noise_type == 'simplex':
                        for i in range(batch_size):
                            base = np.random.uniform(0, 1000)
                            for j in range(LATENT_DIM):
                                x = (base + j % width) / 4
                                y = (base + j / width) / 4
                                my_noise_2[i][j] = ns.snoise2(x, y, octaves=OCTAVES, persistence=PERSISTENCE,
                                                              lacunarity=LACUNARITY)

                    g_loss = g_loss + self.combined.train_on_batch(my_noise_2, valid)
                g_loss = g_loss / rep

                # for _ in range(3):
                # Trénování generátoru (pouze generátor, diskriminátor je zamražen)
                # g_loss = self.combined.train_on_batch(my_noise, valid)

                # Výpis průběhu trénování
                print("%d [Ztráta disk.: %f, přesnost: %.2f%%] [Ztráta gen.: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss))

                # Ukládání generovaných obrázků v pravidelných intervalech
                if epoch % sample_interval == 0:
                    self.sample_images(epoch, noise_type)

        except Exception as e:
            print("Při trénování došlo k chybě,, máš dobře adresu k datasetu? :")
            traceback.print_exc()

    def sample_images(self, epoch,UI=False, noise_type='random', mode='train'):
        """
        Metoda pro vzorkování a uložení vygenerovaných obrázků.

        Parametry:
        - epoch: Číslo epochy
        - iteration: Číslo iterace
        - noise_type: Typ šumu pro generátor (random/perlin/simplex)
        """
        try:
            x, y = 3, 4  # Grid size

            gen_imgs = np.empty((x * y, self.img_rows, self.img_cols, self.channels))
            #noise_imgs = np.empty((x * y, self.img_rows, self.img_cols, self.channels))
            noise=[]

            noise = np.random.normal(0, 1, (x * y, self.LATENT_DIM))
            for i in range(x):
                for j in range(y):
                    seed = np.random.randint(0, 1000)

                    if noise_type == 'perlin':
                        perlin_noise = np.empty((1, self.LATENT_DIM))
                        for k in range(self.LATENT_DIM):
                            xs = (seed + k % WIDTH) / 4
                            ys = (seed + k / WIDTH) / 4
                            perlin_noise[0, k] = ns.pnoise2(xs, ys, octaves=OCTAVES, persistence=PERSISTENCE,
                                                            lacunarity=LACUNARITY)
                        noise = perlin_noise

                    elif noise_type == 'perlin2D':
                        perlin_noise = np.empty((1, self.LATENT_DIM, 3))
                        for k in range(self.LATENT_DIM):
                            perlin_noise[0, k] = ns.pnoise2(i / x, j / y, base=seed)
                        noise = np.reshape(perlin_noise, (1, WIDTH, WIDTH, 3))

                    elif noise_type == 'simplex':
                        simplex_noise = np.empty(self.LATENT_DIM)
                        for k in range(self.LATENT_DIM):
                            xs = (seed + k % WIDTH) / 4
                            ys = (seed + k / WIDTH) / 4
                            simplex_noise[k] = ns.snoise2(xs, ys, octaves=OCTAVES, persistence=PERSISTENCE,
                                                             lacunarity=LACUNARITY)
                        noise[i * y + j] = simplex_noise

                gen_imgs = self.generator.predict(noise)
                # gen_imgs[i * y + j] = gen_img[0]


            # Přizpůsobení rozsahu obrázků na 0 - 1 pro tanh
            gen_imgs = (gen_imgs + 1) / 2

            dpi = matplotlib.rcParams['figure.dpi']
            fig, axs = plt.subplots(x, y, figsize=(SIZE / float(dpi) * y, SIZE / float(dpi) * x))

            cnt = 0
            for i in range(x):
                for j in range(y):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].axis('off')
                    cnt += 1

            output_dir = OUTPUT_PATH + noise_type
            os.makedirs(output_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            if UI:
                plt.suptitle("Vygenerovaný výstup")
            fig.savefig(os.path.join(output_dir, "n_%d.png" % epoch), dpi=dpi)

            # plt.close()

            #print(gen_imgs)
            #print(noise)

            if UI:
                image_paths = []
                for filename in os.listdir(DATASET):
                    if filename.endswith(".png"):
                        image_paths.append(os.path.join(DATASET, filename))

                random_image_paths = random.sample(image_paths, x * y)

                for i in range(x):
                    for j in range(y):
                        image_path = random_image_paths.pop()
                        image = plt.imread(image_path)
                        axs[i, j].imshow(image)
                        axs[i, j].axis('off')

                output_dir = os.path.join(OUTPUT_PATH, noise_type)
                os.makedirs(output_dir, exist_ok=True)

                plt.suptitle("Dataset")
                fig.savefig(os.path.join(output_dir, "d_%d.png" % epoch), dpi=dpi)
                # plt.close()



            cnt = 0
            print(cnt)
            for i in range(x):
                for j in range(y):
                    #axs[i, j].imshow(noise_imgs[cnt])
                    axs[i, j].imshow(noise[cnt].reshape((WIDTH, WIDTH)), cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1

            output_dir = OUTPUT_PATH + noise_type
            os.makedirs(output_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            if UI:
                plt.suptitle("Vstupní šum")
            fig.savefig(os.path.join(output_dir, "o_%d.png" % epoch), dpi=dpi)

            plt.close()




        except Exception as e:
            print("Při vzorkování obrázků došlo k chybě:")
            traceback.print_exc()
        if mode != 'generate' and epoch % EPOCH_TO_SAVE_WEIGHT == 0:
            # Uložení vah
            weights_dir = WEIGHT_PATH + noise_type + '/'
            os.makedirs(weights_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            weights_path_generator = os.path.join(weights_dir, 'weights_g%d.h5' % epoch)
            weights_path_discriminator = os.path.join(weights_dir, 'weights_d%d.h5' % epoch)

            self.generator.save_weights(weights_path_generator)

            self.discriminator.save_weights(weights_path_discriminator)

    # def load_weights(self, weights_file):
    #    self.combined.load_weights(weights_file)


if __name__ == '__main__':

    # Typ šumu: random, perlin noise, simplex noise (random/perlin/simplex)
    noise_type = 'simplex'

    # Režim: trénovací / generovací (train/generate)
    mode = 'train'

    gan = GAN(noise_type)

    if mode == 'train':
        gan.train(epochs=200000, batch_size=BATCH_SIZE, sample_interval=1000, noise_type=noise_type)

    elif mode == 'train_from_stop':
        choosed_epoch = 40
        weights_dir = WEIGHT_PATH + noise_type + '/'
        weights_path_generator = os.path.join(weights_dir, 'weights_g%d.h5' % choosed_epoch)
        weights_path_discriminator = os.path.join(weights_dir, 'weights_d%d.h5' % choosed_epoch)

        gan.generator.load_weights(weights_path_generator)
        gan.discriminator.load_weights(weights_path_discriminator)
        gan.train(epochs=200000, batch_size=BATCH_SIZE, sample_interval=100, noise_type=noise_type,
                  epoch=choosed_epoch + 1)

    elif mode == 'generate':

        choosed_epoch = 17000
        weights_dir = WEIGHT_PATH + noise_type + '/'
        weights_path_generator = os.path.join(weights_dir, 'weights_g%d.h5' % choosed_epoch)
        gan.generator.load_weights(weights_path_generator)

        plot_model(gan.generator, to_file="generator.png", expand_nested=True, dpi=200, show_layer_activations=True,
                   show_shapes=True, show_dtype=True)
        plot_model(gan.discriminator, to_file="discriminator.png", expand_nested=True, dpi=200,
                   show_layer_activations=True, show_shapes=True, show_dtype=True)

        # Generování
        epoch_number = 17000
        noise_type = noise_type

        gan.sample_images(epoch_number, noise_type, mode='generate')

