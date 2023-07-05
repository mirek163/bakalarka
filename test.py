
from __future__ import print_function, division  # Pro kompatibilitu s Python 2
import os  # Manipulace se souborovým systémem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import traceback  # Výpis trasování chyb
from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose  # Vrstvy modelu Keras
from keras.layers import BatchNormalization, LeakyReLU, Conv2D
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

import math
import random
from itertools import product



DATASET = '/home/basamiro/bakalarka/region/'
LATENT_DIM = 100

def smoothstep(t):
    """Smooth curve with a zero derivative at 0 and 1, making it useful for interpolating."""
    return t * t * (3. - 2. * t)


def lerp(t, a, b):
    """Linear interpolation between a and b, given a fraction t."""
    return a + t * (b - a)


class PerlinNoiseFactory(object):
    """Callable that produces Perlin noise for an arbitrary point in an arbitrary number of dimensions.
    The underlying grid is aligned with the integers.
    There is no limit to the coordinates used; new gradients are generated on the fly as necessary.
    """

    def __init__(self, dimension, octaves=1, tile=(), unbias=False):
        """Create a new Perlin noise factory in the given number of dimensions,
        which should be an integer and at least 1.
        More octaves create a foggier and more-detailed noise pattern.  More than 4 octaves is rather excessive.
        ``tile`` can be used to make a seamlessly tiling pattern.  For example:
            pnf = PerlinNoiseFactory(2, tile=(0, 3))
        This will produce noise that tiles every 3 units vertically, but never tiles horizontally.
        If ``unbias`` is true, the smoothstep function will be applied to the output before returning it,
        to counteract some of Perlin noise's significant bias towards the center of its output range.
        """
        self.dimension = dimension
        self.octaves = octaves
        self.tile = tile + (0,) * dimension
        self.unbias = unbias
        self.scale_factor = 2 * dimension ** -0.5
        self.gradient = {}

    def _generate_gradient(self):
        if self.dimension == 1:
            return (random.uniform(-1, 1),)
        random_point = [random.gauss(0, 1) for _ in range(self.dimension)]
        scale = sum(n * n for n in random_point) ** -0.5
        return tuple(coord * scale for coord in random_point)

    def get_plain_noise(self, *point):
        if len(point) != self.dimension:
            raise ValueError("Expected {} values, got {}".format(self.dimension, len(point)))

        grid_coords = []
        for coord in point:
            min_coord = math.floor(coord)
            max_coord = min_coord + 1
            grid_coords.append((min_coord, max_coord))

        dots = []
        for grid_point in product(*grid_coords):
            if grid_point not in self.gradient:
                self.gradient[grid_point] = self._generate_gradient()
            gradient = self.gradient[grid_point]
            dot = 0
            for i in range(self.dimension):
                dot += gradient[i] * (point[i] - grid_point[i])
            dots.append(dot)

        dim = self.dimension
        while len(dots) > 1:
            dim -= 1
            s = smoothstep(point[dim] - grid_coords[dim][0])
            next_dots = []
            while dots:
                next_dots.append(lerp(s, dots.pop(0), dots.pop(0)))
            dots = next_dots

        return dots[0] * self.scale_factor

    def __call__(self, *point):
        ret = 0
        for o in range(self.octaves):
            o2 = 1 << o
            new_point = []
            for i, coord in enumerate(point):
                coord *= o2
                if self.tile[i]:
                    coord %= self.tile[i] * o2
                new_point.append(coord)
            ret += self.get_plain_noise(*new_point) / o2

        ret /= 2 - 2 ** (1 - self.octaves)

        if self.unbias:
            r = (ret + 1) / 2
            for _ in range(int(self.octaves / 2 + 0.5)):
                r = smoothstep(r)
            ret = r * 2 - 1

        return ret


# ----------------Github https://gist.github.com/eevee/26f547457522755cb1fb8739d0ea89a1 ---------------
def perlin_noise(shape, scale=10, octaves=6, persistence=0.5, lacunarity=2):
    """
    Generuje Perlinův šum.

    Parametry:
    - shape: Tuple určující tvar výstupního pole šumu
    - scale: Měřítko určující frekvenci šumu
    - octaves: Počet oktáv nebo úrovní detailu šumu
    - persistence: Parametr persistence řídící amplitudu každého oktávu
    - lacunarity: Parametr lacunarity řídící zvýšení frekvence mezi oktávami

    Vrací:
    - 2D numpy pole s Perlinovým šumem
    """
    rows, cols = shape
    noise = np.zeros(shape)
    pnf = PerlinNoiseFactory(2, octaves=octaves)

    for i in range(rows):
        for j in range(cols):
            amplitude = 1
            frequency = 1
            noise_value = 0

            for _ in range(octaves):
                sample_i = i / scale * frequency
                sample_j = j / scale * frequency

                value = pnf(sample_i, sample_j)
                noise_value += value * amplitude

                amplitude *= persistence
                frequency *= lacunarity

            noise[i, j] = noise_value

    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    return noise


# ---------------------------
# SIMPLEX NOISE



def simplex_noise(shape, scale=10, octaves=6, persistence=0.5, lacunarity=2):
    rows, cols = shape
    noise = np.zeros(shape)

    for i in range(rows):
        for j in range(cols):
            amplitude = 1
            frequency = 1
            noise_value = 0


            for _ in range(octaves):
                seed_1 = np.random.randint(0, 10000) / 10000
                seed_2 = np.random.randint(0, 10000) / 10000
                base_seed = np.random.randint(0, 10000)

                sample_i = i / scale * frequency
                sample_j = j / scale * frequency
                value = ns.snoise2(sample_i + seed_1, sample_j + seed_2, base=base_seed)

                # value = ns.snoise2(seed_1,seed_2, base=base_seed)


                noise_value += value * amplitude
                amplitude *= persistence
                frequency *= lacunarity

            noise[i, j] = noise_value

    return noise




class GAN():

    def __init__(self, noise_type='random'):
        # Velikost vstupního obrázku
        self.img_rows = 256
        self.img_cols = 256
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
        model.add(Dense(256 * 4 * 4, input_dim=self.LATENT_DIM))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(
            Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))  # Upsampling and output

        model.summary()

        noise = Input(shape=(self.LATENT_DIM,))
        img = model(noise)

        return Model(noise, img)


    def build_discriminator(self):
        model = Sequential()

        # Discriminator layers
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
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
            image_files = glob.glob(self.DATASET +'*.png')
            X_train = []
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
                    my_noise = perlin_noise((batch_size, self.LATENT_DIM), scale=10, octaves=6,
                                            persistence=0.5, lacunarity=2.0)
                elif noise_type == 'simplex':
                    my_noise = simplex_noise((batch_size, self.LATENT_DIM), scale=10, octaves=6,
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
                    my_noise = perlin_noise((batch_size, self.LATENT_DIM), scale=scale, octaves=6,
                                            persistence=0.5, lacunarity=2.0)
                elif noise_type == 'simplex':
                    scale = 10.0
                    my_noise = simplex_noise((batch_size, self.LATENT_DIM), scale=scale, octaves=6,
                                             persistence=0.5, lacunarity=2)

                else:
                    my_noise = np.random.normal(0, 1, (batch_size, self.LATENT_DIM))

                for _ in range(3):
                    # Trénování generátoru (pouze generátor, diskriminátor je zamražen)
                    g_loss = self.combined.train_on_batch(my_noise, valid)

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
            # seed = np.random.randint(0, 10000)

            if noise_type == 'perlin':
                my_noise = perlin_noise((x * y, self.LATENT_DIM))
                noise = my_noise

            elif noise_type == 'simplex':
                my_noise = simplex_noise((x * y, self.LATENT_DIM))
                noise = my_noise

            print(noise)

            gen_imgs = self.generator.predict(noise)

            # Přizpůsobení rozsahu obrázků na 0 - 1 pro tanh
            gen_imgs = (gen_imgs+1 ) /2

            fig, axs = plt.subplots(x, y)
            cnt = 0
            for i in range(x):
                for j in range(y):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1

            output_dir = "/home/basamiro/bakalarka/output/ " +noise_type
            os.makedirs(output_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            fig.savefig(os.path.join(output_dir, "n_%d.png" % epoch))
            plt.close()

        except Exception as e:
            print("Při vzorkování obrázků došlo k chybě:")
            traceback.print_exc()
        if mode!= 'generate':
            # Uložení vah
            weights_dir = '/home/basamiro/bakalarka/weight/ ' +noise_type +'/'
            os.makedirs(weights_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            weights_path = os.path.join(weights_dir, 'weights%d.h5' %epoch)
            self.generator.save_weights(weights_path)

    # def load_weights(self, weights_file):
    #    self.combined.load_weights(weights_file)


if __name__ == '__main__':

    # Typ šumu: random, perlin noise, simplex noise (random/perlin/simplex)
    # noise_type = 'random'
    noise_type = 'random'

    # Režim: trénovací / generovací (train/generate)
    # mode = 'train'
    mode = 'train'

    gan = GAN(noise_type)

    if mode == 'train':
        gan.train(epochs=1000001, batch_size=100, sample_interval=10000, noise_type=noise_type)

    elif mode == 'generate':
        # Načtení vah
        weights_path = '/home/basamiro/bakalarka/weight/' + noise_type + '/weights.h5'
        # gan.build_generator(noise_type)
        gan.generator.load_weights(weights_path)

        # Generování
        epoch_number = 100000  # Počet epoch
        noise_type = noise_type  # Typ šumu

        gan.sample_images(epoch_number, noise_type, mode='generate')

