from __future__ import print_function, division  # Pro kompatibilitu s Python 2
import os  # Manipulace se souborovým systémem
import traceback  # Výpis trasování chyb
from keras.layers import Input, Dense, Reshape, Flatten  # Vrstvy modelu Keras
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential, Model  # Modely Keras
from keras.optimizers import Adam  # Optimalizátor Adam
import matplotlib.pyplot as plt  # Vykreslování obrázků
import glob  # Vyhledávání souborů v adresáři
from PIL import Image  # Práce s obrázky
import noise as ns  # Generování šumu
import numpy as np  # Matematické operace s poli


class GAN():

    def __init__(self, noise_type='random'):
        # Velikost vstupního obrázku
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Dimenze latentního vektoru
        self.latent_dim = 100
        self.noise_type = noise_type

        optimizer = Adam(0.0002, 0.5)

        # Sestavení a kompilace diskriminátoru
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Sestavení generátoru
        self.generator = self.build_generator(noise_type=self.noise_type)

        # Generátor přijímá šum jako vstup a generuje obrázky
        z = Input(shape=(self.latent_dim,))
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
        """
        Metoda pro vytvoření a sestavení generátoru modelu.

        Parametry:
        - noise_type: Typ šumu pro generátor (random/perlin/simplex)
        """
        model = Sequential()  # Vytvoření sekvenčního modelu Keras

        # Vrstvy generátoru
        model.add(Dense(256, input_dim=self.latent_dim))  # Plně propojená vrstva s 256 jednotkami
        model.add(LeakyReLU(alpha=0.2))  # Aktivační funkce LeakyReLU s parametrem alpha=0.2
        model.add(BatchNormalization(momentum=0.8))  # Normalizace dávkou s parametrem momentum=0.8
        model.add(Dense(512))  # Plně propojená vrstva s 512 jednotkami
        model.add(LeakyReLU(alpha=0.2))  # Aktivační funkce LeakyReLU s parametrem alpha=0.2
        model.add(BatchNormalization(momentum=0.8))  # Normalizace dávkou s parametrem momentum=0.8
        model.add(Dense(1024))  # Plně propojená vrstva s 1024 jednotkami
        model.add(LeakyReLU(alpha=0.2))  # Aktivační funkce LeakyReLU s parametrem alpha=0.2
        model.add(BatchNormalization(momentum=0.8))  # Normalizace dávkou s parametrem momentum=0.8
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))  # Plně propojená vrstva s aktivací tanh
        model.add(Reshape(self.img_shape))  # Změna tvaru na velikost obrázku
        model.summary()  # Výpis shrnutí modelu

        noise = Input(shape=(self.latent_dim,))  # Vstup pro šum
        # if noise_type == 'perlin':
        #     perlin_noise = np.empty((self.latent_dim,))
        #     for i in range(self.latent_dim):
        #         perlin_noise[i] = ns.pnoise2(i, 0.1)  # Generování Perlinova šumu
        #     noise = Input(shape=(self.latent_dim,), tensor=tf.convert_to_tensor(perlin_noise.reshape((1, -1)), dtype=np.float32))
        # elif noise_type == 'simplex':
        #     simplex_noise = np.empty((self.latent_dim,))
        #     for i in range(self.latent_dim):
        #         simplex_noise[i] = ns.snoise2(i, 0.1)  # Generování simplexního šumu
        #     noise = Input(shape=(self.latent_dim,), tensor=tf.convert_to_tensor(simplex_noise.reshape((1, -1)), dtype=np.float32) )# Změna tvaru na (1, latent_dim)
        # else:
        #     noise = noise  # Vstupní šum

        img = model(noise)  # Generování obrázku pomocí modelu
        return Model(noise, img)  # Vytvoření konečného modelu s vstupem noise a výstupem img

    def build_discriminator(self):
        """
        Metoda pro vytvoření a sestavení diskriminátoru modelu.
        """
        model = Sequential()  # Vytvoření sekvenčního modelu Keras

        # Vrstvy diskriminátoru
        model.add(Flatten(input_shape=self.img_shape))  # Plošná vrstva pro rovnání vstupního obrazu
        model.add(Dense(512))  # Plně propojená vrstva s 512 jednotkami
        model.add(LeakyReLU(alpha=0.2))  # Aktivační funkce LeakyReLU s parametrem alpha=0.2
        model.add(Dense(256))  # Plně propojená vrstva s 256 jednotkami
        model.add(LeakyReLU(alpha=0.2))  # Aktivační funkce LeakyReLU s parametrem alpha=0.2
        model.add(Dense(1, activation='sigmoid'))  # Plně propojená vrstva s aktivací sigmoid

        model.summary()  # Výpis shrnutí modelu

        img = Input(shape=self.img_shape)  # Vstup pro obraz
        validity = model(img)  # Výstup diskriminátoru

        return Model(img, validity)  # Vytvoření modelu s vstupem img a výstupem validity

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
        try:
            # Načtení obrázků ze složky
            image_files = glob.glob("data/input/crack/*.jpg")
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
                    image = image.resize((64, 64))
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
                if noise_type == 'perlin':
                    scale = 10.0
                    myNoise = np.zeros((batch_size, self.latent_dim))
                    for i in range(batch_size):
                        x = np.random.uniform(0, 1)
                        y = np.random.uniform(0, 1)
                        myNoise[i] = ns.pnoise2(x, y, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024,
                                                repeaty=1024, base=0) * scale
                elif noise_type == 'simplex':
                    scale = 10.0
                    myNoise = np.random.uniform(-scale, scale, (batch_size, self.latent_dim))
                else:
                    myNoise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generování obrázků pomocí generátoru
                gen_imgs = self.generator.predict(myNoise)

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
                    myNoise = np.zeros((batch_size, self.latent_dim))
                    for i in range(batch_size):
                        x = np.random.uniform(0, 1)
                        y = np.random.uniform(0, 1)
                        myNoise[i] = ns.pnoise2(x, y, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024,
                                                repeaty=1024, base=0) * scale
                elif noise_type == 'simplex':
                    scale = 10.0
                    myNoise = np.random.uniform(-scale, scale, (batch_size, self.latent_dim))
                else:
                    myNoise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                for _ in range(3):
                    # Trénování generátoru (pouze generátor, diskriminátor je zamražen)
                    g_loss = self.combined.train_on_batch(myNoise, valid)

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
            noise = np.random.normal(0, 1, (x * y, self.latent_dim))

            if noise_type == 'perlin':
                perlin_noise = np.empty((x * y, self.latent_dim))
                for i in range(x * y):
                    for j in range(self.latent_dim):
                        perlin_noise[i, j] = ns.pnoise2(i / (x * y), j / (x * y))
                noise = perlin_noise

            elif noise_type == 'simplex':
                simplex_noise = np.empty((x * y, self.latent_dim))
                for i in range(x * y):
                    for j in range(self.latent_dim):
                        simplex_noise[i, j] = ns.snoise2(i, j)
                noise = simplex_noise

            gen_imgs = self.generator.predict(noise)

            # Přizpůsobení rozsahu obrázků na 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(x, y)
            cnt = 0
            for i in range(x):
                for j in range(y):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                    axs[i, j].axis('off')
                    cnt += 1

            output_dir = "data/output/" + noise_type
            os.makedirs(output_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            fig.savefig(os.path.join(output_dir, "%d.png" % epoch))
            plt.close()

        except Exception as e:
            print("Při vzorkování obrázků došlo k chybě:")
            traceback.print_exc()
        if mode != 'generate':
            # Uložení vah
            weights_dir = 'data/weights/' + noise_type + '/'
            os.makedirs(weights_dir, exist_ok=True)  # Vytvoření výstupního adresáře, pokud neexistuje
            weights_path = os.path.join(weights_dir, 'weights%d.h5' % epoch)
            self.generator.save_weights(weights_path)

    # def load_weights(self, weights_file):
    #    self.combined.load_weights(weights_file)


if __name__ == '__main__':

    # Typ šumu: random, perlin noise, simplex noise (random/perlin/simplex)
    # noise_type = 'random'
    noise_type = 'perlin'

    # Režim: trénovací / generovací (train/generate)
    # mode = 'train'
    mode = 'generate'

    gan = GAN(noise_type)

    if mode == 'train':
        gan.train(epochs=10000, batch_size=32, sample_interval=20, noise_type=noise_type)

    elif mode == 'generate':
        # Načtení vah
        weights_path = 'data/weights/' + noise_type + '/weights.h5'
        # gan.build_generator(noise_type)
        gan.generator.load_weights(weights_path)

        # Generování
        epoch_number = 100000  # Počet epoch
        noise_type = noise_type  # Typ šumu

        gan.sample_images(epoch_number, noise_type, mode='generate')
