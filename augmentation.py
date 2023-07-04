import rasterio
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

image_path = "data/NE1_HR_LC_SR_W_DR.tif" # 21600x10800



x_start = 16200
x_end = 18200

y_start = 2000
y_end = 4000

number_of_pictures = 1
number_of_pictures=number_of_pictures // 32

brightness_factor1 = 1.15
brightness_factor2 = 0.85

contrast_factor1 = 1.15
contrast_factor2 = 0.85


#----------------------------------------

# Open the image using rasterio protože Image size (233280000 pixels) exceeds limit of 178956970 pixels
with rasterio.open(image_path) as dataset:
    image_array = dataset.read()

cropped_image = image_array[:, y_start:y_end, x_start:x_end]
cropped_image = cropped_image.transpose(1, 2, 0)
cropped_image = np.clip(cropped_image, 0, 255).astype(np.uint8)
crop_image = Image.fromarray(cropped_image)
crop_image.save("data/input/c_region.png")


def brightness(image):
    im = ImageEnhance.Brightness(image)
    im_output1 = im.enhance(brightness_factor1)

    im = ImageEnhance.Brightness(image)
    im_output2 = im.enhance(brightness_factor2)
    return im_output1, im_output2

def contrast(image):
    im = ImageEnhance.Contrast(image)
    im_output1 = im.enhance(contrast_factor1)

    im = ImageEnhance.Contrast(image)
    im_output2 = im.enhance(contrast_factor2)
    return im_output1, im_output2


for number in range(number_of_pictures):
    height, width = cropped_image.shape[:2]
    x = np.random.randint(0, width - 64)
    y = np.random.randint(0, height - 64)

    number_of_pictures = number_of_pictures


    random_region = cropped_image[y:y+64, x:x+64]
    random_region_image = Image.fromarray(random_region)
    random_region_path = 'data/input/region/region_'+str(number)+'.png'
    random_region_image.save(random_region_path)

    im_brightness = brightness(random_region_image)
    im_brightness[0].save('data/input/region/region_b_'+str(number)+'.png')
    im_brightness[1].save('data/input/region/region_bb_'+str(number)+'.png')

    im_contrast = contrast(random_region_image)
    im_contrast[0].save('data/input/region/region_c_'+str(number)+'.png')
    im_contrast[1].save('data/input/region/region_cc_'+str(number)+'.png')


    for angle in [90, 180, 270]:
        rotated_region = Image.open(random_region_path)
        rotated_region = rotated_region.rotate(angle, expand=True)
        rotated_region.save('data/input/region/region'+str(angle)+'_'+str(number)+'.png')

        im_brightness = brightness(rotated_region)
        im_brightness[0].save('data/input/region/region_b_'+str(angle)+'_'+str(number)+'.png')
        im_brightness[1].save('data/input/region/region_bb_'+str(angle)+'_'+str(number)+'.png')

        im_contrast = contrast(rotated_region)
        im_contrast[0].save('data/input/region/region_c_'+str(angle)+'_'+str(number)+'.png')
        im_contrast[1].save('data/input/region/region_c_'+str(angle)+'_'+str(number)+'.png')



        im_flip = ImageOps.flip(rotated_region)
        im_flip.save('data/input/region/region' + str(angle) + '_f_' + str(number) + '.png')

        im_brightness = brightness(im_flip)
        im_brightness[0].save('data/input/region/region_b_'+str(angle)+'_f_'+str(number)+'.png')
        im_brightness[1].save('data/input/region/region_bb_'+str(angle)+'_f_'+str(number)+'.png')

        im_brightness = contrast(im_flip)
        im_contrast[0].save('data/input/region/region_c_'+str(angle)+'_f_'+str(number)+'.png')
        im_contrast[1].save('data/input/region/region_cc_'+str(angle)+'_f_'+str(number)+'.png')









