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

from utils import AIPS_module as ai
from utils import AIPS_functions as af
from utils import AIPS_granularity as ag
from utils import AIPS_file_display as afd


def calculate_granularity(image, path, table, classLabel):
    AIPS = ai.Segmentation(Image_name=image, path=path, ch_=table['seed-channeLselect'][0],
                           rmv_object_nuc=table['rmv_object_nuc'][0],
                           block_size=table['block_size'][0],
                           offset=table['offset'][0],
                           clean=3)
    img = AIPS.imageMatrix()
    seedContainer = AIPS.seedSegmentation()
    targetCobntainer = AIPS.cytosolSegmentation(ch2_=table['target-channeLselect'][0],
                                                block_size_cyto=table['block_size_cyto'][0],
                                                offset_cyto=table['offset_cyto'][0],
                                                global_ther=table['global_ther'][0],
                                                rmv_object_cyto=table['rmv_object_cyto'][0],
                                                rmv_object_cyto_small=table['rmv_object_cyto_small'][0],
                                                remove_borders=True)
    seed, target = seedContainer['sort_mask'], targetCobntainer['cseg_mask']
    gran = ag.GRANULARITY(image=img[1], mask=target)
    granData = gran.loopLabelimage(start_kernel=2, end_karnel=50, kernel_size=50)
    granOriginal, _ = gran.featuresTable(features=['label', 'centroid'])
    granData["classLabel"] = classLabel
    targetMerge = afd.Compsite_display(input_image=img[1], mask_roi=target).draw_ROI_contour(channel=1)
    return granOriginal, granData, targetMerge


if (__name__ == "__main__"):
    import argparse

    parser = argparse.ArgumentParser(description='sbatch test')
    parser.add_argument('--file', dest='file', type=str, required=True,
                        help="The name of the bash file to analyze")

    parser.add_argument('--path', dest='path', type=str, required=True,
                        help="The path to the file")

    parser.add_argument('--pathImage', dest='pathImage', type=str, required=True,
                        help="path for merge Image is saved")

    parser.add_argument('--pathTable', dest='pathTable', type=str, required=True,
                        help="path for Table is saved")

    parser.add_argument('--cLass', dest='cLass', type=str, required=True,
                        help="class label")

    args = parser.parse_args()

    Original_path = '/data/kanferg/Images/Stress_Granules_classification/AIPS_dash_final_old-main'
    parametersTable = pd.read_csv(os.path.join(Original_path, 'u2os_42c_e1xy02.csv'))
    parametersTable.head(5)
    parametersTable['global_ther'] = 0.6
    parametersTable['block_size_cyto'] = 89
    parametersTable['rmv_object_cyto_small'] = 0.2

    dfOriginal, dfGranularity, compsite = calculate_granularity(image=args.file, path=args.path, table=parametersTable,
                                                                classLabel=args.cLass)
    image_name = str(args.file)
    image_name = image_name.split('.tif')[0]
    dfGranularity.to_csv(os.path.join(args.path, image_name + '_granularity.csv'))
    dfOriginal.to_csv(os.path.join(args.pathTable, image_name + '_original.csv'))
    plt.imsave(os.path.join(args.pathImage, image_name + '_merge.png'), compsite)