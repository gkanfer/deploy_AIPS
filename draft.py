import time, os, sys
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tfi
import random
os.chdir(r'F:\HAB_2\PrinzScreen\Deploy')
from utils import AIPS_cellpose as AC
from utils import AIPS_granularity as ag

file = 'Cropexp001_13DKO_1-1.tif'
path_input = r'F:\HAB_2\PrinzScreen\Deploy\unitest\mix'

path_out = path_input

#upload model
AIPS_pose_object = AC.AIPS_cellpose(Image_name = file, path= path_input, model_type="cyto", channels=[0,0])
img = AIPS_pose_object.cellpose_image_load()

# create mask for the entire image
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
tfi.imread(os.path.join(path_input,file))

gran = ag.GRANULARITY(image =img,mask = mask)
granData = gran.loopLabelimage(start_kernel = 1, end_karnel = 5, kernel_size=5,deploy=True)
granDataFinal = ag.MERGE().calcDecay(granData,5)

trace_a = -24.440681675906607
trace_b = 31.215097768574996

def classify(n,thold):
    mu = trace_a + trace_b * n
    prob = 1 /(1  + np.exp(-mu))
    return prob, prob > thold
rate = granDataFinal.intensity.values
td = 0.5
prob,prediction = classify(rate,td)

table["predict"] = prob



image_blank = np.zeros_like(img)
binary, table_sel = AIPS_pose_object.call_bin(table_sel_cor = table, threshold = 0.5 ,img_blank = image_blank)

from skimage.draw import disk
#
table_na_rmv_trgt = table.loc[table['predict'] > 0.5, :]
# for z in range(len(table_na_rmv_trgt)):
#     print(z)
#
#
image_blank = np.zeros_like(img)
for z in range(len(table_na_rmv_trgt)-1):
    x, y = table_na_rmv_trgt.loc[table_na_rmv_trgt.index[z],["centroid-0", "centroid-1"]]
    row, col = disk((int(x), int(y)), 20)
    image_blank[row, col] = 1





#
# from skimage import io, filters, measure, color, img_as_ubyte
# from PIL import Image, ImageEnhance, ImageDraw,ImageFont
# import matplotlib.pyplot as plt
# #plt.imshow(binary)
# img_gs = img_as_ubyte(binary)
# # from skimage.measure import label, regionprops
# # img_gs = label(img_gs)
# if os.path.exists(os.path.join(path_out, 'binary.tif')):
#     os.remove(os.path.join(path_out, 'binary.tif'))
# tfi.imsave(os.path.join(path_out, 'binary.tif'), img_gs)
#
# with open(os.path.join(path_out, 'cell_count.txt'), 'r') as f:
#     prev_number = f.readlines()
# new_value = int(prev_number[0]) + len(table_na_rmv)
# with open(os.path.join(path_out, 'cell_count.txt'), 'w') as f:
#     f.write(str(new_value))




