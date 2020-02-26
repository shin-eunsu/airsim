import airsim #pip install airsim
import numpy as np
import os

# for car use CarClient()
client = airsim.CarClient()

responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
response = responses[0]

# get numpy array
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

# reshape array to 4 channel image array H X W X 4
img_rgb = img1d.reshape(response.height, response.width, 3)

# original image is fliped vertically
img_rgb = np.flipud(img_rgb)


# just for fun add little bit of green in all pixels
img_rgb[:, :, 1:2] = 100

# write to png
filename = 'd:/airsim'
airsim.write_png(os.path.normpath(filename + '.png'), img_rgb)
