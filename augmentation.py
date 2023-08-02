import rasterio
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

image_path = "data/input/region.tif" # 21600x10800

cim_path = 'bakalarka'
output_path = 'data/input/region/'

x_start, x_end = 13000, 14000

y_start, y_end = 1500, 2500

number_of_pictures = 100 #*4

brightness_factor1 = 1.15
brightness_factor2 = 0.85

contrast_factor1 = 1.15
contrast_factor2 = 0.85


#----------------------------------------

#Tohle protože velikost obrázku je (233280000 pixelů) a to přesahuje limit 178956970 pixelů
with rasterio.open(image_path) as dataset:
    image_array = dataset.read()

cropped_image = image_array[:, y_start:y_end, x_start:x_end]
cropped_image = cropped_image.transpose(1, 2, 0)
cropped_image = np.clip(cropped_image, 0, 255).astype(np.uint8)
crop_image = Image.fromarray(cropped_image)
crop_image.save(cim_path+"c_region.png")

epoch = 0


def print_status():
    global epoch
    epoch += 1
    print(f"{epoch}/{number_of_pictures * 4}\n----------------------------")

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
    x = np.random.randint(0, width - 256)
    y = np.random.randint(0, height - 256)

    number_of_pictures = number_of_pictures


    random_region = cropped_image[y:y+256, x:x+256]
    random_region_image = Image.fromarray(random_region)
    random_region_path = output_path+'region_'+str(number)+'.png'
    random_region_image.save(random_region_path)
    print_status()

    #im_brightness = brightness(random_region_image)

    #im_brightness[0].save(output_path+'region_b_'+str(number)+'.png')
    #print_status()

    #im_brightness[1].save(output_path+'region_bb_'+str(number)+'.png')
    #print_status()

    #im_contrast = contrast(random_region_image)
    #im_contrast[0].save(output_path+'region_c_'+str(number)+'.png')
    #print_status()

    #im_contrast[1].save(output_path+'region_cc_'+str(number)+'.png')
    #print_status()


    for angle in [90, 180, 270]:
        rotated_region = Image.open(random_region_path)
        rotated_region = rotated_region.rotate(angle, expand=True)
        rotated_region.save(output_path+'region'+str(angle)+'_'+str(number)+'.png')
        print_status()

        #im_brightness = brightness(rotated_region)
        #im_brightness[0].save(output_path+'region_b_'+str(angle)+'_'+str(number)+'.png')
        #print_status()

        #im_brightness[1].save(output_path+'region_bb_'+str(angle)+'_'+str(number)+'.png')
        #print_status()

        #im_contrast = contrast(rotated_region)
        #im_contrast[0].save(output_path+'region_c_'+str(angle)+'_'+str(number)+'.png')
        #print_status()

        #im_contrast[1].save(output_path+'region_c_'+str(angle)+'_'+str(number)+'.png')
        #print_status()

        #im_flip = ImageOps.flip(rotated_region)
        #im_flip.save(output_path+'region' + str(angle) + '_f_' + str(number) + '.png')

        #im_brightness = brightness(im_flip)
        #im_brightness[0].save(output_path+'region_b_'+str(angle)+'_f_'+str(number)+'.png')
        #print_status()

        #im_brightness[1].save(output_path+'region_bb_'+str(angle)+'_f_'+str(number)+'.png')
        #print_status()

        #im_brightness = contrast(im_flip)
        #im_contrast[0].save(output_path+'region_c_'+str(angle)+'_f_'+str(number)+'.png')
        #print_status()

        #im_contrast[1].save(output_path+'region_cc_'+str(angle)+'_f_'+str(number)+'.png')
        #print_status()









