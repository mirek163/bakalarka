from __future__ import print_function, division  # Pro kompatibilitu s Python 2
import os  # Manipulace se souborovým systémem
import traceback  # Výpis trasování chyb
import random

from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, Dropout  # Vrstvy modelu Keras
from keras.layers import BatchNormalization, LeakyReLU, Conv2D
from keras.models import Sequential, Model  # Modely Keras
from keras.optimizers import Adam  # Optimalizátor Adam
import matplotlib.pyplot as plt  # Vykreslování obrázků
import glob  # Vyhledávání souborů v adresáři
from PIL import Image  # Práce s obrázky
# import noise as ns  # Generování šumu
import numpy as np  # Matematické operace s poli

import noises

DATASET = 'data/input/region/'
LATENT_DIM = 100
DATASET_NUMBER = 5024
BATCH_SIZE = 128




class GAN():

    def __init__(self, noise_type='random'):
        # Velikost vstupního obrázku
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Dimenze latentního vektoru
        self.noise_type = noise_type
        self.DATASET = DATASET
        self.LATENT_DIM = LATENT_DIM

        optimizer = Adam(0.0002, 0.5)

        # Sestavení a kompilace diskriminátoru
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Sestavení generátoru
        self.generator = self.build_generator(noise_type=self.noise_type)

        # Generátor přijímá šum jako vstup a generuje obrázky
        z = Input(shape=(self.LATENT_DIM,))
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
        model.add(Dense(256 * 16 * 16, input_dim=self.LATENT_DIM))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((16, 16, 256)))
        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))  # Dropout layer
        model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))  # Dropout layer
        model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))  # Dropout layer
        model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))  # Upsampling and output

        model.summary()

        noise = Input(shape=(self.LATENT_DIM,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        # Discriminator layers
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(256, 256, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))  # Dropout layer
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))  # Dropout layer
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))  # Dropout layer
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=(256, 256, 3))
        validity = model(img)

        return Model(img, validity)
    def train(self, epochs, batch_size=128, sample_interval=50, noise_type='random'):
        """
        Metoda pro trénování modelu.

        Parametry:
        - epochs: Celkový počet epoch (iterací) pro trénování modelu.
        - batch_size: Velikost dávky obrázků použitých při každé iteraci trénování.
        - sample_interval: Počet epoch mezi výstupy generátoru (generované obrázky) pro vizuální kontrolu.
        - noise_type: Typ šumu použitý při generování obrázků. Možné hodnoty jsou 'random' (náhodný šum),
          'perlin' (Perlinův šum) a 'simplex' (simplexový šum).
        """

        ## Najdi soubor s největší váhou
        # weight_files = glob.glob(WEIGHTS_DIRECTORY+'/'+noise_type+'/weights*.h5')
        # if weight_files:
        #    largest_weight_file = max(weight_files, key=lambda x: int(re.search(r'\d+', x).group()))
        #    starting_epoch = int(re.search(r'\d+', largest_weight_file).group()) + 1
        #    self.generator.load_weights(largest_weight_file)
        # else:
        #    starting_epoch = 0

        try:
            # Načtení obrázků ze složky
            image_files = glob.glob(self.DATASET+'*.png')
            X_train = []
            random.shuffle(image_files)
            image_files = image_files[:DATASET_NUMBER]
            for image_file in image_files:
                try:
                    # Otevření obrázku, úprava velikosti a konverze do RGB formátu
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
            X_train = X_train.reshape((-1, self.img_rows, self.img_cols, self.channels))


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

                # if noise_type == 'perlin':
                #     scale = 10.0
                #     my_noise = np.zeros((batch_size, self.LATENT_DIM))
                #     for i in range(batch_size):
                #         x = np.random.uniform(0, 1000)
                #         y = np.random.uniform(0, 1000)
                #         my_noise[i] = ns.pnoise2(x, y, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024,
                #                                 repeaty=1024, base=0) * scale

                if noise_type == 'perlin':
                    my_noise = noises.perlin_noise((batch_size, self.LATENT_DIM), scale=10, octaves=6,
                                                   persistence=0.5, lacunarity=2.0)
                elif noise_type == 'simplex':
                    my_noise = noises.simplex_noise((batch_size, self.LATENT_DIM), scale=10, octaves=6,
                                                    persistence=0.5, lacunarity=2)

                else:
                    my_noise = np.random.normal(0, 1, (batch_size, self.LATENT_DIM))


                # Generování obrázků pomocí generátoru
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

                # Generování šumu pro kombinovaný model

                if noise_type == 'perlin':
                    scale = 10.0
                    my_noise = noises.perlin_noise((batch_size, self.LATENT_DIM), scale=scale, octaves=6,
                                                   persistence=0.5, lacunarity=2.0)
                elif noise_type == 'simplex':
                    scale = 10.0
                    my_noise = noises.simplex_noise((batch_size, self.LATENT_DIM), scale=scale, octaves=6,
                                                    persistence=0.5, lacunarity=2)

                else:
                    my_noise = np.random.normal(0, 1, (batch_size, self.LATENT_DIM))

                g_loss = self.combined.train_on_batch(my_noise, valid)

                # for _ in range(3):
                #    # Trénování generátoru (pouze generátor, diskriminátor je zamražen)
                #    g_loss = self.combined.train_on_batch(my_noise, valid)

                # Výpis průběhu trénování
                print("%d [Ztráta disk.: %f, přesnost: %.2f%%] [Ztráta gen.: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss))

                # Ukládání generovaných obrázků v pravidelných intervalech
                if epoch % sample_interval == 0:
                    self.sample_images(epoch, noise_type)

        except Exception as e:
            print("Při trénování došlo k chybě,, máš dobře adresu k datasetu? :")
            traceback.print_exc()

    def sample_images(self, epoch, noise_type='random', mode='train'):
        """
        Metoda pro vzorkování a uložení vygenerovaných obrázků.

        Parametry:
        - epoch: Číslo epochy
        - iteration: Číslo iterace
        - noise_type: Typ šumu pro generátor (random/perlin/simplex)
        """
        try:
            x, y = 3, 4  # V jakém gridu se budou obrázky generovat

            noise = np.random.normal(0, 1, (x * y, self.LATENT_DIM))
            #seed = np.random.randint(0, 10000)

            if noise_type == 'perlin':
                perlin_noise = noises.perlin_noise((x * y, self.LATENT_DIM))
                noise = perlin_noise

            elif noise_type == 'simplex':
                simplex_noise = noises.simplex_noise((x * y, self.LATENT_DIM))
                noise = simplex_noise

            print(noise)

            # původní noise
            if mode == 'generate':
                fig, axs = plt.subplots(x, y)
                cnt = 0
                for i in range(x):
                    for j in range(y):
                        axs[i, j].imshow(
                            noise[cnt, :].reshape((int(np.sqrt(self.LATENT_DIM)), int(np.sqrt(self.LATENT_DIM)))),
                            cmap='gray')
                        axs[i, j].axis('off')
                        cnt += 1

                output_dir = "data/output/" + noise_type
                os.makedirs(output_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
                fig.savefig(os.path.join(output_dir, "o_%d.png" % epoch))
                plt.close()

            gen_imgs = self.generator.predict(noise)

            # Přizpůsobení rozsahu obrázků na 0 - 1 pro tanh
            gen_imgs = (gen_imgs+1)/2

            fig, axs = plt.subplots(x, y)
            cnt = 0
            for i in range(x):
                for j in range(y):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1

            output_dir = "data/output/" + noise_type
            os.makedirs(output_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            fig.savefig(os.path.join(output_dir, "n_%d.png" % epoch))
            plt.close()

        except Exception as e:
            print("Při vzorkování obrázků došlo k chybě:")
            traceback.print_exc()
        if mode != 'generate' and epoch != 0: # nechci generovat 0 epochu
            # Uložení vah
            weights_dir = 'data/weights/' + noise_type + '/'
            os.makedirs(weights_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            weights_path = os.path.join(weights_dir, 'weights%d.h5' %epoch)
            self.generator.save_weights(weights_path)

    #def load_weights(self, weights_file):
    #    self.combined.load_weights(weights_file)

if __name__ == '__main__':

    # Typ šumu: random, perlin noise, simplex noise (random/perlin/simplex)
    # noise_type = 'random'
    noise_type = 'perlin'

    # Režim: trénovací / generovací (train/generate)
    # mode = 'train'
    mode = 'train'

    gan = GAN(noise_type)

    if mode == 'train':
        gan.train(epochs=10000, batch_size=BATCH_SIZE, sample_interval=1000, noise_type=noise_type)

    elif mode == 'generate':
        # Načtení vah
        weights_path = 'data/weights/'+noise_type+'/weights.h5'
        # gan.build_generator(noise_type)
        gan.generator.load_weights(weights_path)

        # Generování
        epoch_number = 100000  # Počet epoch
        noise_type = noise_type  # Typ šumu

        gan.sample_images(epoch_number, noise_type,mode='generate')

