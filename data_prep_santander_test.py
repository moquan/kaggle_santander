
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

# Get training header list:
input_list_file_name = work_dir + 'input_list.cpk'
load_from_file = False
if load_from_file:
  input_list = cPickle.load(open(input_list_file_name, 'rb'))
else:
  prices_raw = pd.read_csv(work_dir + 'train.csv')
  meta_data = prices_raw.drop(['ID', 'target'], axis=1)
  input_list = list(meta_data)
  prices = copy.deepcopy(prices_raw)
  for item in input_list:
      prices[item] = prices[item].replace(0,np.NaN)
  prices = prices.dropna(axis=1, how='all')
  meta_data = prices.drop(['ID', 'target'], axis=1)
  input_list = list(meta_data)
  cPickle.dump(input_list, open(input_list_file_name, 'wb'))
  
prices_raw = pd.read_csv(work_dir + 'test.csv')

# Remove previous all 0 columns
prices = prices_raw[input_list]
IDs    = prices_raw['ID']
test_ID_file_name=work_dir+"test_ID.dat"
cPickle.dump(IDs, open(test_ID_file_name, 'wb'))

    
prices_binary = copy.deepcopy(prices)
for item in input_list:
    # prices_binary[item] = prices_binary[item].replace(np.NaN, 0)
    prices_binary[item+'new'] = np.where(prices_binary[item] > 0, 1, 0)


print prices_binary.shape

input_list_new = []
input_list_new += input_list
for item in input_list:
    input_list_new.append(item+'new')
print len(input_list_new)

X_test = np.array(prices_binary[input_list_new].values)
# y_train = np.array(prices_binary['target'].values)
data = {"x_test": X_test,
          # "y_train": y_train,
}

data['x_test'].shape

min_max_file_name=work_dir+"min_max.dat"
data_norm_file_name=work_dir+"data_norm_test.dat"
load_from_file=True

if not load_from_file:
    min_max = {
      "x_min": np.amin(data['x_train'], axis=0).reshape(-1),
      "x_max": np.amax(data['x_train'], axis=0).reshape(-1),
      "y_min": np.amin(data['y_train']),
      "y_max": np.amax(data['y_train']),
    }
    cPickle.dump(min_max, open(min_max_file_name, 'wb'))
else:
    min_max = cPickle.load(open(min_max_file_name, 'rb'))

data_norm = {}
x_diff = (min_max["x_max"] - min_max["x_min"])
# y_diff = (min_max["y_max"] - min_max["y_min"])
data_norm["x_test"] = (data["x_test"] - min_max["x_min"][None,:]) / x_diff[None,:] * 2. - 1.
# data_norm["y_train"] = (data["y_train"] - min_max["y_min"]) / y_diff * 2. - 1.

cPickle.dump(data_norm, open(data_norm_file_name, 'wb'))
