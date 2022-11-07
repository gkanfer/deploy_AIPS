'''
function description:
Binary byes' classification

Input table should be two classified tables e.g.: normal vs phenotype
Paramters:
1) prior flags: optional
2)
'''

import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
import seaborn as sns
#sns.set()
import arviz as az
import pymc3 as pm
print(pm.__version__)
import theano.tensor as tt
import patsy

import os
import re
import glob
import random
# import plotnine
from sklearn import preprocessing
from tqdm import tqdm

import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from skimage import measure, restoration,morphology
from skimage import io, filters, measure, color, img_as_ubyte
from skimage.draw import disk
from skimage import measure, restoration,morphology

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)

from utils import AIPS_module as ai
from utils import AIPS_functions as af
from utils import AIPS_granularity as ag
from utils.AIPS_cellpose import granularityMesure_cellpose
from utils.Baysian_training import bayesModelTraining

# Make granularity Table per cell
#
#
# outPath = r'F:\HAB_2\PrinzScreen\Deploy\unitest'
#
# NormPath = r'F:\HAB_2\PrinzScreen\Deploy\unitest\normImage'
# NormFile = 'Cropexp001_WT_3-3.tif'
# granularityMesure_cellpose(file =NormFile, path = NormPath, clean = 1500, classLabel = 0,outPath = outPath, outputTableName = 'norm.csv')
#
# phenoPath = r'F:\HAB_2\PrinzScreen\Deploy\unitest\phenoImage'
# phenoFile = 'Cropexp001_PEX3KO7_4-2.tif'
# granularityMesure_cellpose(file =phenoFile, path = phenoPath, clean = 1500, classLabel = 1,outPath = outPath, outputTableName = 'pheno.csv')


pathIn =  r'F:\HAB_2\PrinzScreen\Deploy\unitest'
pathOut =  r'F:\HAB_2\PrinzScreen\Deploy\unitest\output'
files = glob.glob(pathname=pathIn+"\*.csv")
bayesModelTraining(files = files,kernelSize = 5,pathOut = pathOut, reportFile = "kernel20unitest")


#
# #dfMergeFinel = pd.read_csv(os.path.join(path_input,file))
# dfMergeFinel = ag.MERGE().mergeTable(tableInput_name_list = files)
# dfMergeFinelFitelrd = ag.MERGE().calcDecay(dfMergeFinel,5)
#
# from matplotlib.backends.backend_pdf import PdfPages
#
# def generate_plots():
#     def line():
#         dfline = pd.DataFrame({"kernel": dfMergeFinel.kernel.values, "Signal intensity (ratio)": dfMergeFinel.intensity.values, "class":dfMergeFinel.classLabel.values})
#         fig, ax = plt.subplots()
#         sns.lineplot(data=dfline, x="kernel", y="Signal intensity (ratio)",hue="class").set(title='Granularity spectrum plot')
#         return ax
#
#     def plotBox():
#         classLabel = dfMergeFinelFitelrd.classLabel.values.tolist()
#         intensity = dfMergeFinelFitelrd.intensity.values.tolist()
#         df = pd.DataFrame({"classLabel": classLabel, "intensity": intensity})
#         fig, ax = plt.subplots()
#         sns.boxplot(data=df,x="classLabel", y="intensity").set(title='Cell area distribution')
#         return ax
#     plot1 = plotBox()
#     plot2 = line()
#     return (plot1, plot2)
#
#
# def plots2pdf(plots, fname):
#     with PdfPages(fname) as pp:
#         for plot in plots:
#             pp.savefig(plot.figure)
#
# plots2pdf(generate_plots(), os.path.join(path_input,'preTrainingPlots.pdf'))
#
#
# rate = dfMergeFinelFitelrd.intensity.values
# y_0 = dfMergeFinelFitelrd.classLabel.values
# with pm.Model() as model_logistic_basic:
#     a = pm.Normal('a',0,10)
#     b = pm.Normal('b',0,10)
#     mu = a + pm.math.dot(rate,b)
#     theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
#     bd = pm.Deterministic('bd',-a/b)
#     yl = pm.Bernoulli('yl',theta,observed = y_0)
#     trace = pm.sample(4000, tune=4000, target_accept=0.99,random_seed=RANDOM_SEED)
# idata = az.from_pymc3(trace, model=model_logistic_basic)
#
# idx = np.argsort(rate)
# def classify(n,thold):
#     mu = trace['a'].mean() + trace['b'].mean() * n
#     prob = 1 /(1  + np.exp(-mu))
#     return prob, prob > thold
# rate = dfMergeFinelFitelrd.intensity.values
# td = 0.5
# prob,prediction = classify(rate,td)
# y_true = y_0
# y_pred = np.where(prediction==True,1,0)
# performance = precision_recall_fscore_support(y_true, y_pred, average='macro')
#
#
# from matplotlib.backends.backend_pdf import PdfPages
# path_input = r'F:\HAB_2\PrinzScreen\Deploy\unitest'
# with PdfPages(os.path.join(path_input,'multipage_pdf.pdf')) as pdf:
#     idx = np.argsort(rate)
#     plt.figure(figsize=(3, 3))
#     plt.plot(rate[idx],theta[idx],color ='b',lw=3)
#     plt.axvline(trace['bd'].mean(),ymax=1,color = 'r')
#     bd_hdi = pm.hdi(trace['bd'])
#     plt.fill_betweenx([0,1],bd_hdi[0],bd_hdi[1], color = 'r')
#     plt.plot(rate,y_0,'o',color = 'k')
#     pdf.savefig()
#     plt.close()
#
#     plt.figure(figsize=(3, 3))
#     az.plot_trace(idata, var_names=('theta'))
#     pdf.savefig()
#     plt.close()
#
#     plt.figure(figsize=(3, 3))
#     confusion_matrix = metrics.confusion_matrix(np.array(dfMergeFinelFitelrd.classLabel.values, dtype=int),
#                                                 np.where(prediction, 1, 0))
#     cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
#     cm_display.plot()
#     plt.text(0.01, 0.5, "Precision :{}".format(performance[0]), fontsize=10, transform=plt.gcf().transFigure)
#     plt.text(0.01, 0.4, "Recall :{}".format(performance[1]), fontsize=10, transform=plt.gcf().transFigure)
#     plt.text(0.01, 0.4, "F1 score :{}".format(performance[2]), fontsize=10, transform=plt.gcf().transFigure)
#     pdf.savefig()
#     plt.close()
#
#
#
#
# axes = az.plot_trace(idata,var_names=('theta'))
# fig = axes.ravel()[0].figure
# fig.savefig( os.path.join(path_input,'test.png'))
#
# def generate_plots():
#     def a():
#         fig, ax = plt.subplots()
#         az.plot_trace(idata,var_names=('theta'))
#         return ax
#     def b():
#         fig, ax = plt.subplots()
#         az.plot_trace(idata,var_names=('theta'))
#         return ax
#     def theta():
#         fig, ax = plt.subplots()
#         az.plot_trace(idata,var_names=('theta'))
#         return ax
#     # def border():
#     #     idx = np.argsort(rate)
#     #     fig, ax = plt.subplots()
#     #     ax.plot(rate[idx],theta[idx],color ='b',lw=3)
#     #     ax.axvline(trace['bd'].mean(),ymax=1,color = 'r')
#     #     bd_hdi = pm.hdi(trace['bd'])
#     #     ax.fill_betweenx([0,1],bd_hdi[0],bd_hdi[1], color = 'r')
#     #     ax.plot(rate,y_0,'o',color = 'k')
#     #     return ax
#
#
#     plot1 = a()
#     plot2 = b()
#     plot3 = theta()
#     #plot4 = border()
#     return (plot1, plot2,plot3)
#
#
# def plots2pdf(plots, fname):
#     with PdfPages(fname) as pp:
#         for plot in plots:
#             pp.savefig(plot.figure)
# plots2pdf(generate_plots(), os.path.join(path_input,'preformancePlots.pdf'))
#
#
#
# plt.figure(figsize=(3, 3))
# plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
# plt.title('Page One')
# pdf.savefig()  # saves the current figure into a pdf page
# plt.close()
#
#
#
# from matplotlib.backends.backend_pdf import PdfPages
# plt.figure(figsize=(3, 3))
# az.plot_trace(idata,var_names=('theta'))
#
#
#
#
#
#
# theta = trace['theta'].mean(0)
# idx = np.argsort(rate)
# plt.plot(rate[idx],theta[idx],color ='b',lw=3)
# plt.axvline(trace['bd'].mean(),ymax=1,color = 'r')
# bd_hdi = pm.hdi(trace['bd'])
# plt.fill_betweenx([0,1],bd_hdi[0],bd_hdi[1], color = 'r')
# plt.plot(rate,y_0,'o',color = 'k')
#
#
#     idx = np.argsort(rate)
#     fig, ax = plt.subplots()
#     ax.plot(rate[idx], theta[idx], color='b', lw=3)
#     ax.axvline(trace['bd'].mean(), ymax=1, color='r')
#     bd_hdi = pm.hdi(trace['bd'])
#     ax.fill_betweenx([0, 1], bd_hdi[0], bd_hdi[1], color='r')
#     ax.plot(rate, y_0, 'o', color='k')
#     return ax
