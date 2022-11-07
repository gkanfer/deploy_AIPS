'''
function description:
1) cell segmented using cellpose
2) clean data based on cell area detected
3) granularity measure
output:
1) Segmented image composition with area label
2) Area histogram plot
3) Segmented image composition after removal of small objects
4) plot of Mean intensity over opening operation (granularity spectrum)

example

path_input = r'F:\HAB_2\PrinzScreen\Deploy\testData'
pathOut = r'F:\HAB_2\PrinzScreen\Deploy\out'
file = 'crop_exp001_13DKO_2-1.tif'
'''

from utils.AIPS_cellpose import granularityMesure_cellpose

if (__name__ == "__main__"):
    import argparse
    parser = argparse.ArgumentParser(description='measure granularity from segmented images')
    parser.add_argument('--file', dest='file', type=str, required=True,
                        help="Image name")
    parser.add_argument('--path', dest='path', type=str, required=True,
                        help="The path to the image")
    parser.add_argument('--clean', dest='clean', type=int, required=False,
                        help="Remove object bellow the area")
    parser.add_argument('--classLabel', dest='classLabel', type=int, required=True,
                        help="phenotype label for classification, integer")
    parser.add_argument('--pathOut', dest='pathOut', type=str, required=True,
                        help="path to save images and plots")
    parser.add_argument('--outputTableName', dest='outputTableName', type=str, required=True,
                        help="Name of the table output e.g. outputTableNorm.csv")
    args = parser.parse_args()

    granularityMesure_cellpose(file = args.file, path = args.path, clean = args.clean, classLabel = args.classLabel,outPath = args.pathOut, outputTableName = args.outputTableName)











