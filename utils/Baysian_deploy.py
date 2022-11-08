import time, os, sys
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tfi
import random
from skimage import io, filters, measure, color, img_as_ubyte
import string
import random

from utils import AIPS_cellpose as AC
from utils import AIPS_granularity as ag
from utils import AIPS_file_display as afd


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def BayesianGranularityDeploy(file,path,kernel_size,trace_a,trace_b,thold,pathOut):
    '''
    on the fly cell call function for activating cells

    :param file: str, single channel target image
    :param path: str
    :param kernel_size: int,
    :param trace_a: int,
    :param trace_b: int
    :param thold: int, probability threshold for calling cells
    :param pathOut: str
    :return: binary mask for activating the called cell
    '''
    AIPS_pose_object = AC.AIPS_cellpose(Image_name=file, path=path, model_type="cyto", channels=[0, 0])
    img = AIPS_pose_object.cellpose_image_load()
    # create mask for the entire image
    mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
    gran = ag.GRANULARITY(image=img, mask=mask)
    granData = gran.loopLabelimage(start_kernel=1, end_karnel=kernel_size, kernel_size=kernel_size)
    granDataFinal = ag.MERGE().calcDecay(granData, kernel_size)
    def classify(n, thold):
        mu = trace_a + trace_b * n
        prob = 1 / (1 + np.exp(-mu))
        return prob, prob > thold
    rate = granDataFinal.intensity.values
    prob, prediction = classify(rate, thold)
    table["predict"] = prob
    image_blank = np.zeros_like(img)
    binary, table_sel = AIPS_pose_object.call_bin(table_sel_cor=table, threshold=0.9, img_blank=image_blank)
    img_gs = img_as_ubyte(binary)
    if os.path.exists(os.path.join(pathOut, 'binary.tif')):
        os.remove(os.path.join(pathOut, 'binary.tif'))
    tfi.imsave(os.path.join(pathOut, 'binary.tif'), img_gs)
    with open(os.path.join(pathOut, 'cell_count.txt'), 'r') as f:
        prev_number = f.readlines()
    new_value = int(prev_number[0]) + len(table_sel)
    with open(os.path.join(pathOut, 'cell_count.txt'), 'w') as f:
        f.write(str(new_value))


def BayesianGranularityDeployTest(file,path,kernel_size,trace_a,trace_b,thold,pathOut):
    '''
    on the fly cell call function for activating cells

    :param file: str, single channel target image
    :param path: str
    :param kernel_size: int,
    :param trace_a: int,
    :param trace_b: int
    :param thold: int, probability threshold for calling cells
    :param pathOut: str
    :return: binary mask for activating the called cell
    '''
    AIPS_pose_object = AC.AIPS_cellpose(Image_name=file, path=path, model_type="cyto", channels=[0, 0])
    img = AIPS_pose_object.cellpose_image_load()
    # create mask for the entire image
    mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
    gran = ag.GRANULARITY(image=img, mask=mask)
    granData = gran.loopLabelimage(start_kernel=1, end_karnel=kernel_size, kernel_size=kernel_size)
    granDataFinal = ag.MERGE().calcDecay(granData, kernel_size)
    def classify(n, thold):
        mu = trace_a + trace_b * n
        prob = 1 / (1 + np.exp(-mu))
        return prob, prob > thold
    rate = granDataFinal.intensity.values
    prob, prediction = classify(rate, thold)
    table["predict"] = prob
    image_blank = np.zeros_like(img)
    binary, table_sel = AIPS_pose_object.call_bin(table_sel_cor=table, threshold=0.9, img_blank=image_blank)
    return img, mask, table, binary, table_sel




