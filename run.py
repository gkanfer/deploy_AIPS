'''
@echo on
call activate D:\Gil\anaconda_gil\envs\cellpose
call python D:\Gil\AIPS\deployPeroxisiome\deploy_AIPS\run.py
@pause
'''

from utils.Baysian_deploy import BayesianGranularityDeploy
file = 'input.tif'
path_input = r'D:\Gil\AIPS\deployPeroxisiome\deployActiveFolder'
path_out = path_input

BayesianGranularityDeploy(file = file, path = path_input, kernel_size = 5, trace_a = -27, trace_b = 33, thold = 0.9, pathOut = path_out)

