import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.spatial import distance_matrix
from scipy import stats

def class_iden(df,y):
    """
    Identify majority class and minority class for binary class data

    Parameters
    ----------
    data : dataframe
        an input dataframe
        class_col : str
          a class column of data

    Returns:
        minor_class : a minority class of data
        major_class : a majority class of data
    """

    d = df[y].value_counts().to_dict()
    d_key = list(d.keys())

    all_equal = True
    for k in d_key[1:]:
      if d[d_key[0]] != d[k]:
        all_equal = False
        break

    if all_equal:
      return d_key[0], d_key[1]

    return min(d,key = d.get), max(d,key = d.get)

def class_count(data,class_col):
    minor_c, major_c = class_iden(data, class_col)
    maj_no = data[data[class_col] == major_c].shape[0]
    min_no = data[data[class_col] == minor_c].shape[0]
    return maj_no,min_no

def data_dist(data,class_col, fig_name):
    """
    Show scatter plot for each class (Only for 2 attributes data).

    The maximum class for plot is 14.

    Parameters
    ----------
    data : dataframe
        an input dataframe
    class_col : str
        a class column of data

    Returns
    -------
    None
    """
    minor_c, major_c = class_iden(data, class_col)
    data_color = ['#FF0000' if c==major_c else '#00FF00' for c in data[class_col]]
    plt.scatter(data['Att1'],data['Att2'],color=data_color)
    plt.xlim([-2,2.5])
    plt.ylim([-1.2,2])
    plt.savefig(fig_name+'.png')
    plt.show()

def sampling(data,s_size):
    """
    Random samples from the data with size s_size

    Parameters
    ----------
    data : dataframe
        a dataframe to do sampling
    s_size : int
        number of samples to sampling

    Returns
    -------
    data.iloc[rand_list] : dataframe
        a dataframe from sampling data
    """
    rand_list = [random.randint(0,len(data)-1) for _ in range(s_size)]
    return data.iloc[rand_list]

def bootstrap(data,s_size,b_size):
    """
    Random samples from the data with size s_size for b_size times

    Parameters
    ----------
    data : dataframe
        a dataframe to do sampling
    s_size : int
        number of samples to sampling
    b_size : int
        number of bootstrapping

    Returns
    -------
    [sampling(data,s_size) for _ in range(b_size)] : list
        a list of dataframe which are from sampling data
    """
    return [sampling(data,s_size) for _ in range(b_size)]

def euclid_distance(p1,p2):
  return np.sqrt(np.sum(np.square(p1 - p2)))

def Neighborhood(arr):
    """
    Compare the pairwise distances between two elements and count the number

    of neighborhood (points that the distance is lower than or equal to the interesting point)

    including the interesting point

    Ex arr = [2,3,6,1,8] -> nbh = [2,3,4,1,5]

    Parameters
    ----------
    arr : numpy array
        a 1-dimensional array which each element are distance

    Returns
    -------
    nbh : numpy array
        an array which each element is the number of neighborhood of that point
    """
    arr_size = len(arr)
    n_arr = arr.reshape((1,arr_size))
    distance_diff = n_arr-n_arr.T
    nbh = np.sum(np.where(distance_diff>=0,1,0),axis=0)
    return nbh

def NBH_Matrix(data):
    """
    Compute the pairwise distances between two points and find the number of neighborhood

    of point q respect to point p for all pair (p,q)

    Ex arr = [[906 892]
              [870 323]
              [433 480]
              [602 695]
              [569 849]]

      Neighbor_matrix =  [[1 4 5 3 2]
                          [4 1 3 2 5]
                          [5 4 1 2 3]
                          [4 5 3 1 2]
                          [3 5 4 2 1]]

    Parameters
    ----------
    data : numpy array
        a 2-dimensional array which each row is a data point

    Returns
    -------
    Neighbor_matrix : numpy array
        a matrix which element [i,j] represent the number of neighborhood of point i respect to point j
    """
    d_size = len(data)
    dist_matrix = distance_matrix(data,data)
    # dist_matrix = distance_matrix(data,data)
    # print(dist_matrix)
    Neighbor_matrix = np.ones((d_size,d_size))
    Neighbor_matrix = np.apply_along_axis(Neighborhood, 1, dist_matrix)
    return Neighbor_matrix

def MassRatio(data):
    """
    Compute the mass ratio of all pairwise data points

    Parameters
    ----------
    data : numpy array
        a 2-dimensional array which each row is a data point

    Returns
    -------
    Neighbor_matrix : numpy array
        mass ratio of all pairwise data points
    """
    minor_NBH_matrix = NBH_Matrix(data)
    # print(minor_NBH_matrix)
    return minor_NBH_matrix/np.transpose(minor_NBH_matrix)

def MOF_p(pre_arr,i):
  arr = np.delete(pre_arr,i,axis=0)
  return np.var(arr)

def MOF(data):
    """
    Compute the mass ratio variance of all data points

    Parameters
    ----------
    data : numpy array
        a 2-dimensional array which each row is a data point

    Returns
    -------
    MRV_matrix : numpy array
        an array of mass ratio variance of all data points
    """
    MR_Matrix = MassRatio(data)
    d_size = len(data)
    MRV_Matrix = np.zeros(d_size)
    for i in range(d_size):
      MRV_Matrix[i] = MOF_p(MR_Matrix[:,i],i)

    return MRV_Matrix

def minmax_normalization(arr):
  '''
  Calculate minmax normalization of data
  '''
  # find min and max of data
  max_data = np.amax(arr, axis=0)
  min_data = np.amin(arr, axis=0)

  # normalize data
  n_arr = (arr-min_data)/(max_data-min_data)

  # replace nan data with 0
  n_arr[np.isnan(n_arr)] = 0

  return n_arr

def n_partition(data, class_col, n=4, normalize = None):
  '''
  Compute MOF, sort data and divide data in to n partition, then return list of index of data in each partition.
  The first list has the lowest MOF

  Input
  - n : number of partition. n=4 refers to quartile partition, n=10 refers to decile partition.
  - normalize : normalize method before calculate MOF
    - None : do not normalize data
    - 'Zscore' : use zscore of data to calculate MOF ((x-mean)/sd)
    - 'MinMax' : Scale data into range [0,1]. ((x-x_min)/(x_max-x_min))
  '''

  if normalize == 'Zscore':
    # Normalize data
    ndata = stats.zscore(data.drop([class_col], axis=1), axis=0, ddof=1)
    # replace nan data with 0
    ndata[np.isnan(ndata)] = 0

  elif normalize == 'MinMax':
    # Normalize data
    ndata = minmax_normalization(data.drop([class_col], axis=1))
    # replace nan data with 0
    ndata[np.isnan(ndata)] = 0

  else:
    ndata = data.drop([class_col], axis=1)

  # calculate MOF and sort
  data_size = len(ndata)
  MOF_list = MOF(ndata)
  d_size = len(ndata)
  MOF_sort = np.argsort(MOF_list).tolist()
  # print(MOF_sort)

  # create empty partition collection
  partition_collection = []
  start_pos, end_pos = 0, int(data_size/n)
  for i in range(1,n+1):
    if i==n:
      end_pos = data_size

    partition_index = MOF_sort[start_pos:end_pos]
    partition_collection.append(data.iloc[partition_index])

    start_pos, end_pos = end_pos, int((i+1)*data_size/n)

  return partition_collection

def QPBT(data,class_col, s_size, b_size, npartition=4, normalize_method=None):
  '''
  Quartile-pattern bootstrapping.
  '''
  # define minor class and major class
  min_class, maj_class = class_iden(data, class_col)
  col_name = data.columns.values.tolist()

  # create bootstrap collection
  bootstrap_collection = []

  d_size = len(data)
  # divide data according to class
  minor_df = data[data[class_col]==min_class].reset_index(drop=True)
  major_df = data[data[class_col]==maj_class].reset_index(drop=True)

  # divide minority data into npartition groups
  partition_data = n_partition(minor_df, class_col, n=npartition, normalize=normalize_method)

  # Do bootstrapping
  for i in range(b_size):
    #create empty dataframe
    sampling_df = pd.DataFrame(columns = col_name)

    # add all upper half of data to sampling df
    for j in range(int(npartition/2),npartition):

      sampling_df = pd.concat([sampling_df, partition_data[j]])

    # bootstrap each of rest partition and add to sampling_df
    for j in range(int(npartition/2)):

      sampling_df = pd.concat([sampling_df, sampling(partition_data[j],len(partition_data[j]))])

    # bootstrap major data and do sampling_df
    sampling_df = pd.concat([sampling_df, sampling(major_df,len(major_df))])

    # ignore index
    sampling_df = sampling_df.reset_index(drop = True)
    sampling_df = sampling_df.apply(pd.to_numeric)

    # add sampling_df to bootstrap_collection
    bootstrap_collection.append(sampling_df)

  return bootstrap_collection