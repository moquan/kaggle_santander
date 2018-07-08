
# coding: utf-8

# In[1]:


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


# In[2]:


work_dir = '/Users/tiannan/Santander_Hackathon/'
prices_raw = pd.read_csv(work_dir + 'train.csv')


# In[5]:


prices_raw.head(3)


# In[6]:


meta_data = prices_raw.drop(['ID', 'target'], axis=1)
input_list = list(meta_data)
prices_raw.head(3)


# In[7]:


prices = copy.deepcopy(prices_raw)
for item in input_list:
    prices[item] = prices[item].replace(0,np.NaN)
    
    


# In[8]:


prices.head(3)


# In[9]:


prices = prices.dropna(axis=1, how='all')
prices.head()
meta_data = prices.drop(['ID', 'target'], axis=1)
input_list = list(meta_data)


# In[12]:


print len(input_list)
prices.head(3)


# In[18]:


prices_binary = copy.deepcopy(prices)
for item in input_list:
    prices_binary[item] = prices_binary[item].replace(np.NaN, 0)
    prices_binary[item+'new'] = np.where(prices_binary[item] > 0, 1, 0)


# In[19]:


print prices_binary.shape
prices_binary.head()


# In[20]:


input_list_new = []
input_list_new += input_list
for item in input_list:
    input_list_new.append(item+'new')
print len(input_list_new)


# In[21]:


X_train = np.array(prices_binary[input_list_new].values)
y_train = np.array(prices_binary['target'].values)
data = {"x_train": X_train,
          "y_train": y_train,
}


# In[22]:


data['x_train'].shape


# In[23]:


data['y_train'].shape


# In[25]:


min_max_file_name=work_dir+"min_max.dat"
data_norm_file_name=work_dir+"data_norm.dat"
load_from_file=False

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

