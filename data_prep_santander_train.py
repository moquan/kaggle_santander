# data_prep.py
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style="whitegrid", color_codes=True)
import cPickle

work_dir = "/home/dawna/tts/mw545/DVExp/hh/"

def load_data(input_csv_file_name=work_dir+"Prices_raw.csv"):
  # load data from csv file (stratpy not available)
  prices_raw = pd.read_csv(input_csv_file_name)
  ''' 
  TODO: 
  1. Make a list of input column names, for easier use in the future
    Basically, remove "target" and "ID"
  2. replace all the 0 with NaN, using something like this:
      prices = prices.replace(0,np.NaN)
    Run a for loop for all input columns?
  3. remove input columns that has 0 only, using something like this:
      b = prices.describe()
      b['count']
    Also, return the remaining input dimension, and modify input_list
  '''
  return prices, input_list, input_dim

def calculate_prices_binary(prices, input_list, input_dim):
  ''' 
  TODO:
  1. For inputs, Replace all the 0 with NaN
  2. Double the number of input columns, such that 
    the first D dimensions are original data, 
    and the next D dimensions are if the value is 0; -1 if it is 0, 1 if it is non-zero
    think of a good column name for those binary columns, e.g. original_name+"_bin"
    * Worth checking if the original names are of same length
  3. Replace all the NaN back to 0
  '''
  return prices_binary, input_list, input_dim

def make_train_test_data(prices_binary, input_list, input_dim):
  ''' 
  TODO:
  Make a list of the name of input columns
  e.g.
    features = ['deltaBid','deltaAsk','deltaMid','midToPaid','midToGiven','bidSize1','askSize1','bidToPaid','askToGiven','bidToGiven','askToPaid', 'book_pressure','spread', 'day_of_week','time_in_sec', 'bid2','ask2','bidSize2','askSize2','deltaBidSize2','deltaAskSize2']
  But this example is hard-coding; for our task, use all the columns except "target"
  '''
  # Split x and y by column name
  X_train = np.array(prices_binary[features].values)
  y_train = np.array(prices_binary['target'].values)
  data = {"x_train": X_train,
          "y_train": y_train,
  }
  return data

def min_max_norm(data, min_max_file_name=work_dir+"min_max.dat", data_norm_file_name=work_dir+"data_norm.dat", load_from_file=False):
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
  y_diff = (min_max["y_max"] - min_max["y_min"])
  data_norm["x_train"] = (data["x_train"] - min_max["x_min"][None,:]) / x_diff[None,:] * 2. - 1.
  data_norm["y_train"] = (data["y_train"] - min_max["y_min"]) / y_diff * 2. - 1.

  cPickle.dump(data_norm, open(data_norm_file_name, 'wb'))
  return data_norm

def data_prep(input_csv_file_name=work_dir+"train.csv", min_max_file_name=work_dir+"min_max.dat", 
      data_norm_file_name=work_dir+"data_norm.dat"):

  prices, input_list, input_dim = load_data(input_csv_file_name=input_csv_file_name)
  prices_binary, input_list, input_dim = calculate_prices_binary(prices, input_list, input_dim)
  data = make_train_test_data(prices_binary, input_list, input_dim)
  data_norm = min_max_norm(data, min_max_file_name, data_norm_file_name)


if __name__ == '__main__': 
  data_prep()

  import cPickle
  import numpy as np
  min_max_file_name=work_dir+"min_max.dat"
  min_max = cPickle.load(open(min_max_file_name, 'rb'))

  data_norm_file_name=work_dir+"data_norm.dat"
  data_norm = cPickle.load(open(data_norm_file_name, 'rb'))
  print data_norm["x_train"].shape
  # print data_norm["x_test"].shape
  print data_norm["y_train"].shape
  # print data_norm["y_test"].shape
  # print data_norm["z_train"].shape
  # print data_norm["z_test"].shape


