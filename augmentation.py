import rasterio
import numpy as np
from PIL import Image
image_path = "data/NE1_HR_LC_SR_W_DR.tif" # 21600x10800



x_start = 16200
x_end = 18200

y_start = 2000
y_end = 4000

number_of_pictures = 20

#----------------------------------------

# Open the image using rasterio protože Image size (233280000 pixels) exceeds limit of 178956970 pixels
with rasterio.open(image_path) as dataset:
    image_array = dataset.read()

cropped_image = image_array[:, y_start:y_end, x_start:x_end]
cropped_image = cropped_image.transpose(1, 2, 0)
cropped_image = np.clip(cropped_image, 0, 255).astype(np.uint8)
crop_image = Image.fromarray(cropped_image)
crop_image.save("data/input/c_region.png")

number_of_pictures = number_of_pictures //5
for number in range(number_of_pictures):
    height, width = cropped_image.shape[:2]
    x = np.random.randint(0, width - 64)
    y = np.random.randint(0, height - 64)

    random_region = cropped_image[y:y+64, x:x+64]

    random_region_image = Image.fromarray(random_region)
    random_region_path = 'data/input/region/region_'+str(number)+'.png'
    random_region_image.save(random_region_path)

    for angle in [0, 90, 180, 270]:
        rotated_region = Image.open(random_region_path)
        rotated_region = rotated_region.rotate(angle, expand=True)
        rotated_region.save('data/input/region/region'+str(angle)+'_'+str(number)+'.png')
