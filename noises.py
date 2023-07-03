import numpy as np

import math
import random
from itertools import product

import noise as ns
# PERLIN NOISE
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
#----------------Github https://gist.github.com/eevee/26f547457522755cb1fb8739d0ea89a1 ---------------
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

#---------------------------
#SIMPLEX NOISE



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
                value = ns.snoise2(sample_i + seed_1, sample_j + seed_2, base=base_seed) # u tohohle použiju funkci, protože se v tom chci ještě  vyznat

                #value = ns.snoise2(seed_1,seed_2, base=base_seed)


                noise_value += value * amplitude
                amplitude *= persistence
                frequency *= lacunarity

            noise[i, j] = noise_value

    return noise
