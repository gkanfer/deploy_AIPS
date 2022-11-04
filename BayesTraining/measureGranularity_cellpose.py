import matplotlib.pyplot as plt
import tifffile as tfi
import numpy as np
from PIL import Image
import plotly.express as px
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
from skimage import io, filters, measure, color, img_as_ubyte
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
from skimage.exposure import rescale_intensity
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import base64
from datetime import datetime
import timeit
import glob
import seaborn as sns

from utils import AIPS_module as ai
from utils import AIPS_functions as af
from utils import AIPS_granularity as ag
from utils import AIPS_file_display as afd
from utils import AIPS_cellpose as AC


path_input = r'F:\HAB_2\PrinzScreen\Deploy\testData'
file = 'exp001_13DKO_2-1.tif'


AIPS_pose_object = AC.AIPS_cellpose(Image_name = file, path= path_input, model_type="cyto", channels=[0,0])
img = AIPS_pose_object.cellpose_image_load()
# create mask for the entire image
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
# area image
sns.histplot(data = table, x = 'area', kde = True,color = sns.color_palette("Set2")[1],binwidth = 500)
plt.savefig('save_as_a_png.png')
# save image mask counter
dipObject = afd.Compsite_display(input_image=img[0,200:700,200:700], mask_roi=mask)
counter = dipObject.draw_ROI_contour(channel=None)

PIL_image = dipObject.display_image_label( table = table, font_select = "arial.ttf", font_size = 24)
plt.imshow(PIL_image)