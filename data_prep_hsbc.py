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

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100

from_date = '2017.09.15'
to_date = '2017.09.30'
start_time = '03:00'
end_time = '15:00'

sym = '`USDINR'
site = "`LOH"

work_dir = "/home/dawna/tts/mw545/DVExp/hh/"

def load_data(input_csv_file_name=work_dir+"Prices_raw.csv"):
  # load data from csv file (stratpy not available)
  prices_raw = pd.read_csv(input_csv_file_name)
  # parse timestamps correctly
  for t in [u'date' , u'ebsMarketUpdateTime', u'feedHandlerPublishTime', u'feedHandlerReceiveTime', u'eventCaptureTime']:
      prices_raw[t] = pd.to_datetime(prices_raw[t])

  T = [u'date' , u'time' , u'ebsMarketUpdateTime', u'feedHandlerPublishTime', u'feedHandlerReceiveTime', u'eventCaptureTime']
  timeDeltas = prices_raw[T]
  timeDeltas['marketLatency'] = timeDeltas['feedHandlerReceiveTime'] - timeDeltas['ebsMarketUpdateTime']
  timeDeltas['processLatency'] = timeDeltas['feedHandlerPublishTime'] - timeDeltas['feedHandlerReceiveTime']
  timeDeltas[['marketLatency','processLatency',]].describe().T[['mean','std','max']]
  prices = prices_raw[['date','bid','ask','bid2','ask2','bidSize1','askSize1','bidSize2','askSize2','paid', 'given']]
  prices['bid'] = prices['bid'].replace(0,np.NaN)
  prices['ask'] = prices['ask'].replace(0,np.NaN)
  prices['bid2'] = prices['bid2'].replace(0,np.NaN)
  prices['ask2'] = prices['ask2'].replace(0,np.NaN)
  prices['paid'] = prices['paid'].replace(0,np.NaN)
  prices['given'] = prices['given'].replace(0,np.NaN)
  prices['mid'] =  prices['ask']
  prices['mid'] = 0.5*(prices['bid'] + prices['mid'])
  prices.index = prices_raw.feedHandlerReceiveTime
  prices = prices.drop_duplicates()
  return prices

def calculate_prices_delta(prices):
  columns = ['bid','ask','bid2','ask2','bidSize1','askSize1','bidSize2','askSize2','mid']
  prices_delta = prices[columns] - prices[columns].shift(1)
  prices_delta.rename(columns = {'mid':'deltaMid','bid':'deltaBid','ask':'deltaAsk','bidSize1':'deltaBidSize1','askSize1':'deltaAskSize1',
                                'bidSize2':'deltaBidSize2','askSize2':'deltaAskSize2'}, inplace=True)
  # add back old prices, and a midDiff for learning later
  LL = ['mid','bid','ask','bidSize1','bidSize2','askSize1','askSize2']
  prices_delta[LL] = prices[LL]
  prices_delta['midDiffInterval'] = (prices_delta['deltaMid'] != 0).cumsum()
  # drop some features
  # LLL = ['bid2','ask2','bidSize2','askSize2','deltaBidSize2','deltaAskSize2']
  # for l in LLL:
  #     prices_delta.drop(l,1,inplace=True)
  # time feature (on feedHandlerRecieve), date,time ... 
  prices_delta['date'] = prices.date
  prices_delta['day_of_week'] = prices_delta['date'].dt.dayofweek
  prices_delta['time'] = prices.index
  prices_delta['time_in_sec'] = (prices_delta['time']-prices_delta['date']).dt.total_seconds()
  # trade Features, print,tradeSeq,lastPaid,lastGiven,bidToPaid,bidToGiven,midToPaid ...
  atomicTrades = prices[['paid','given']].loc[(prices['paid']>1) | (prices['given']>1)]
  atomicTrades.loc[atomicTrades['paid'] <1, 'paid' ] = np.NaN
  atomicTrades.loc[atomicTrades['given'] <1, 'given' ] = np.NaN
  atomicTrades = atomicTrades.replace(0,np.NaN)
  prices_delta['paid'] = atomicTrades['paid']
  prices_delta['given'] = atomicTrades['given']
  prices_delta['print'] = np.where((prices_delta['paid']>1) | (prices_delta['given']>1),1,0)
  prices_delta['tradeSeq'] = prices_delta['print'].cumsum()
  prices_delta['lastPaid'] = prices_delta['paid'].ffill()
  prices_delta['lastGiven'] = prices_delta['given'].ffill()
  prices_delta.drop('paid',1,inplace=True)
  prices_delta.drop('given',1,inplace=True)
  prices_delta['midToPaid'] = prices_delta['mid'] - prices_delta['lastPaid']
  prices_delta['midToGiven'] = prices_delta['mid'] - prices_delta['lastGiven']
  prices_delta['bidToPaid'] = prices_delta['bid'] - prices_delta['lastPaid']
  prices_delta['bidToGiven'] = prices_delta['bid'] - prices_delta['lastGiven']
  prices_delta['askToPaid'] = prices_delta['ask'] - prices_delta['lastPaid']
  prices_delta['askToGiven'] = prices_delta['ask'] - prices_delta['lastGiven']
  # book preasure feature
  prices['book_pressure'] = prices['mid'] - (prices['bidSize1']*prices['bid'] + prices['askSize1']*prices['ask'])/(prices['bidSize1']+prices['askSize1'])
  prices_delta['book_pressure'] = prices['mid'] - (prices['bidSize1']*prices['bid'] + prices['askSize1']*prices['ask'])/(prices['bidSize1']+prices['askSize1'])
  # spread feature
  prices_delta['spread'] = prices_delta['ask'] - prices_delta['bid']
  # create feature to learn, ie next move (not to be used as covariates!)
  prices_delta['midDiff'] = prices_delta['mid'].diff()
  prices_delta['nextMidDiff'] = prices_delta['midDiff'].shift(-1)
  prices_delta['nextMidVariation'] = prices_delta['nextMidDiff'].replace(to_replace=0, method='bfill')
  prices_delta.dropna(inplace=True)
  prices_delta = prices_delta.replace(0,np.NaN)
  import functions as func
  func.formatdf(prices_delta.describe().transpose())
  prices_delta = prices_delta.replace(np.NaN,0)
  prices_delta_clean = prices_delta[(np.abs(stats.zscore(prices_delta['deltaMid'])) < 5)]
  prices_delta_clean = prices_delta_clean.replace(0,np.NaN)
  prices_delta_clean = prices_delta_clean.replace(np.NaN,0)
  return prices_delta_clean

def make_train_test_data(prices_delta_clean):
  features = ['deltaBid','deltaAsk','deltaMid','midToPaid','midToGiven','bidSize1','askSize1','bidToPaid','askToGiven','bidToGiven','askToPaid', 'book_pressure','spread',
              'day_of_week','time_in_sec', 'bid2','ask2','bidSize2','askSize2','deltaBidSize2','deltaAskSize2']
  OUT = (prices_delta_clean.date == '2017.09.29') | (prices_delta_clean.date == '2017.09.28') 
  OUT = OUT | (prices_delta_clean.date == '2017.09.27') 
  IN = ~OUT
  X_train = np.array(prices_delta_clean[IN][features].values)
  y_train = np.array(prices_delta_clean[IN]['nextMidVariation'].values)
  z_train = np.array(prices_delta_clean[IN]['mid'].values)
  X_test = np.array(prices_delta_clean[OUT][features].values)
  y_test = np.array(prices_delta_clean[OUT]['nextMidVariation'].values)
  z_test = np.array(prices_delta_clean[OUT]['mid'].values)
  y_train[y_train<0] = -1
  y_train[y_train>0] = 1
  y_test[y_test<0] = -1
  y_test[y_test>0] = 1
  data = {"x_train": X_train,
          "y_train": y_train,
          "z_train": z_train,
          "x_test":  X_test,
          "y_test":  y_test,
          "z_test":  z_test
  }
  # print y_test.shape
  return data

def min_max_norm(data, min_max_file_name=work_dir+"min_max.dat", data_norm_file_name=work_dir+"data_norm.dat", load_from_file=False):
  if not load_from_file:
    min_max = {
      "x_min": np.amin(data['x_train'], axis=0).reshape(-1),
      "x_max": np.amax(data['x_train'], axis=0).reshape(-1),
      "y_min": np.amin(data['y_train']),
      "y_max": np.amax(data['y_train']),
      "z_min": np.amin(data['z_train']),
      "z_max": np.amax(data['z_train'])
    }
    cPickle.dump(min_max, open(min_max_file_name, 'wb'))
  else:
    min_max = cPickle.load(open(min_max_file_name, 'rb'))

  data_norm = {}
  x_diff = (min_max["x_max"] - min_max["x_min"])
  y_diff = (min_max["y_max"] - min_max["y_min"])
  z_diff = (min_max["z_max"] - min_max["z_min"])
  data_norm["x_train"] = (data["x_train"] - min_max["x_min"][None,:]) / x_diff[None,:] * 2. - 1.
  data_norm["x_test"]  = (data["x_test"]  - min_max["x_min"][None,:]) / x_diff[None,:] * 2. - 1.
  data_norm["y_train"] = (data["y_train"] - min_max["y_min"]) / y_diff * 2. - 1.
  data_norm["y_test"]  = (data["y_test"]  - min_max["y_min"]) / y_diff * 2. - 1.
  data_norm["z_train"] = (data["z_train"] - min_max["z_min"]) / y_diff * 2. - 1.
  data_norm["z_test"]  = (data["z_test"]  - min_max["z_min"]) / y_diff * 2. - 1.

  cPickle.dump(data_norm, open(data_norm_file_name, 'wb'))
  return data_norm

def data_prep(input_csv_file_name=work_dir+"Prices_raw.csv", min_max_file_name=work_dir+"min_max.dat", 
      data_norm_file_name=work_dir+"data_norm.dat"):

  prices = load_data(input_csv_file_name=input_csv_file_name)
  prices_delta_clean = calculate_prices_delta(prices)
  data = make_train_test_data(prices_delta_clean)
  data_norm = min_max_norm(data, min_max_file_name, data_norm_file_name)


  '''
  Load input_csv_file_name
  Remove useless columns:
    index, sym, siteCode, instrument, status, ebsReferenceTime, ebsMarketUpdateTime, feedHandlerReceiveTime, feedHandlerPublishTime, eventCaptureTime, bidRegular, askRegular, regularSize, numberBidMaker1, numberAskMaker1
  Calculate date as day_in_the_week e.g. Monday=1, Friday=5
    * It might be worth writing a function (see below) that calculates day_in_the_week instead of hard-coding, just in case that the test data has dates in October, for example
    Then, add 5 columns, such that if Monday, the 5 columns are 1,0,0,0,0; Wednesday, 0,0,1,0,0, etc.
      * I guess we don't need 7 columns, since there is no transaction on weekends?
    Finally, remove "date" column
  In columns "paid" and "given", if it is blank, replace with 0
  Then, add 2 columns, name as "paid_bin", "given_bin", that it is binary if "paid" or "given" is non-zero (event detection)
  In "time", convert hh-mm-ss to seconds only
  Add a column, "time_diff", the difference between current "time" and previous "time"; if the row is first of a new day, "time_diff"=0
    Add another column "new_day", that is a binary, if the current row is the first in a new day, "new_day"=1
  Add output columns:
    1. current mid price
    2. next mid price
    3. binary: 1 for increase, 0 for decrease
    4. how many steps later does this occur?

  Expected columns (b: binary, s: seconds, others: float):
    day_in_the_week(b*5), time(s), time_diff(s), new_day(b), bid,ask, paid, given, paid_bin(b), given_bin(b), bid2, bid3, bidSize1, bidSize2, bidSize3,n umberBidMaker2, numberBidMaker3, ask2, ask3, askSize1, askSize2, askSize3, numberAskMaker2, numberAskMaker3, current_mid_price, next_mid_price, up_or_down, how_far_later

  Then perform a min-max normalisation; for each column, find min, find max, then (x-min)/(max-min)
  Save the min and max of each column somewhere, say norm_csv_file_name, with 2 rows only

  Add a new column, "have_data", make it all 1; this is for the case that the test data is too short

  Expected columns (b: binary, s: seconds, others: float):
    day_in_the_week(b*5), time(s), time_diff(s), new_day(b), bid,ask, paid, given, paid_bin(b), given_bin(b), bid2, bid3, bidSize1, bidSize2, bidSize3,n umberBidMaker2, numberBidMaker3, ask2, ask3, askSize1, askSize2, askSize3, numberAskMaker2, numberAskMaker3; current_mid_price, next_mid_price, up_or_down, how_far_later; have_data

  Save to output_csv_file_name

  '''



def draw_N_rows_starting_t(x_file, y_file, t, N):
  '''
  return a matrix of N*D, D is number of columns; the first row is index t; t -- t+N-1
  if not enough e.g. t+N-1 > T: make up the first few rows with zeros, then put true data in the last few rows; 
    This way, "have_data" is 0 for the first few rows
  We might run some testing that, even if the model takes 100 input instances, we input only 60 true data points (short history), and fill the first 40 with 0s
  '''
  return matrix_x

if __name__ == '__main__': 
  data_prep()
  import cPickle
  import numpy as np
  min_max_file_name=work_dir+"min_max.dat"
  min_max = cPickle.load(open(min_max_file_name, 'rb'))

  data_norm_file_name=work_dir+"data_norm.dat"
  data_norm = cPickle.load(open(data_norm_file_name, 'rb'))
  print data_norm["x_train"].shape
  print data_norm["x_test"].shape
  print data_norm["y_train"].shape
  print data_norm["y_test"].shape
  print data_norm["z_train"].shape
  print data_norm["z_test"].shape


