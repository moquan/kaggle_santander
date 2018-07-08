
# coding: utf-8

# data_prep.py
import warnings, copy
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style="whitegrid", color_codes=True)
import cPickle

work_dir = '/home/dawna/tts/mw545/DVExp/hh/'

test_ID_file_name=work_dir+"test_ID.dat"
IDs = cPickle.load(open(test_ID_file_name, 'rb'))
# IDs = prices_raw['ID']

predict_output_file_name = work_dir+"predict_output.dat"
predict_output = cPickle.load(open(predict_output_file_name, 'rb'))

assert IDs.values.shape[0] == predict_output.shape[0]

for i in range(IDs.values.shape[0]):
  IDs[i] ...
  predict_output[i] ...
