import pandas as pd
import numpy as np
import math

class Node(object):
  def __init__(self,name,parent=None,split_val = None):
    self.name = name
    self.parent= parent
    self.children = dict()
    self.split_value = split_val

  def add_child(self,key,child):
    self.children[key]=child

  def print_node(self):
    #print(self.parent.name,':',self.name)
    if self.is_leaf():
      return
    if self.split_value == None:
      for e in self.children:
        print(self.name,':',e,':',self.children[e].name)
        self.children[e].print_node()
    else :
      print(self.name,':','<',self.split_value,':',self.children['left'].name)
      self.children['left'].print_node()
      print(self.name,':','>=',self.split_value,':',self.children['right'].name)
      self.children['right'].print_node()

  def is_leaf(self):
    return len(self.children)==0

def minor_class(df,y):
  d = df[y].value_counts().to_dict()
  return min(d,key = d.get)

def major_class(df,y):
  d = df[y].value_counts().to_dict()
  return max(d,key = d.get)

def minor_major_class(df,y):
  '''
  return minority class and majority class of the data
  '''
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

def entropy(data):
  a = data.value_counts()/data.shape[0]
  ent = np.sum(-a*np.log2(a+1e-9))
  return ent

def cate_entropy(df,attr,y):
  '''
  calculate and return categorical entropy of attribute 'attr' of data 'df'
  '''
  a=df[[attr,y]].value_counts().to_dict()
  # print(a)
  b=df[attr].value_counts().to_dict()
  # print(b)
  c=df[y].unique()
  a_key = a.keys()
  e = sum(b[i]/df.shape[0]*sum(-a[(i,j)]/b[i]*math.log(a[(i,j)]/b[i],2) for j in c if (i,j) in a_key) for i in b.keys())
  return e

def nume_entropy(df,attr,y,candidate):
    d_size = df.shape[0]
    left_dat = df[df[attr]<=candidate][y]
    right_dat = df[df[attr]>candidate][y]
    tot_entropy = left_dat.shape[0]*entropy(left_dat)/d_size + right_dat.shape[0]*entropy(right_dat)/d_size
    return tot_entropy

def nume_best_split_value(df,attr,y):
    '''
        Sort value in attribute and let 10, 20,..., 90 to be candidate
    '''
    nume_list = sorted(df[attr].unique())
    d_size = len(nume_list)
    pos = int(d_size/10)-1
    best_split_val = (nume_list[pos]+nume_list[pos+1])/2
    best_split_entropy = nume_entropy(df,attr,y,best_split_val)
    # print(nume_list)
    # if len(nume_list)==1:
    #   return nume_list[0]

    for i in range(2,10):
      pos = int(i*d_size/10)-1
      candidate = (nume_list[pos]+nume_list[pos+1])/2
      curr_entropy = nume_entropy(df,attr,y,candidate)
      if curr_entropy < best_split_entropy:
          best_split_val = candidate
          best_split_entropy = curr_entropy
    # min_key = min(entropy_dict, key=entropy_dict.get)
    return best_split_val

def best_split_measure(df,y):
    attr_list = df.drop(y,axis=1).columns.tolist()
    # assign initial best split attribute and entropy
    best_split_attr = attr_list[0]
    best_split_entropy = 0
    if df[best_split_attr].dtype=='O':
      best_split_entropy = cate_entropy(df,best_split_attr,y)
    else :
      nume_split = nume_best_split_value(df,best_split_attr,y)
    best_split_entropy = nume_entropy(df,best_split_attr,y,nume_split)

    # find attribute with minimum entropy
    for e in attr_list[1:]:
      if df[e].dtype=='O':
          curr_entropy = cate_entropy(df,e,y)
      else :
          nume_split = nume_best_split_value(df,e,y)
          curr_entropy = nume_entropy(df,e,y,nume_split)

      if curr_entropy < best_split_entropy:
          best_split_attr = e
          best_split_entropy = curr_entropy

    return best_split_attr

def stop_criteria(df,y):
    '''
    stopping criteria condition for recursive partitioning
    '''
    stop = False
    c = df[y].unique()
    if(len(c)==1):
      stop = True
    return stop

def RPA(node,df,y,max_depth = 7, max_features = None):
  '''
  recursive partitioning algorithm for building decision tree
  '''
  # return leaf node if there is only 1 distinct value in class column
  if stop_criteria(df,y) == True:
    c = df[y].unique()[0]
    node.name = c
    return

  # return leaf node if tree reach max depth
  if max_depth == 0:
    class_dict = df[y].value_counts().to_dict()
    # print(class_dict)
    node.name = max(class_dict, key=class_dict.get)
    return

  # drop column with 1 distinct value (except column of class)
  for col in df.drop(y,axis=1).columns:
    # if show:
    # print(str(col)+':'+str(df[col].unique()))
    if len(df[col].unique()) == 1:
        # print('drop')
        df = df.drop(col,axis=1)

  # print(df)
  # print('...........................')
  # if there is no attribute left, then return leaf node with majority
  if len(df.columns)==1:
    # print('no attribute\n')
    class_dict = df[y].value_counts().to_dict()
    node.name = max(class_dict, key=class_dict.get)
    return

  # define number of features to consider
  # print(isinstance(max_features, float))
  if isinstance(max_features, float):
    features_size = int(max_features*(df.shape[1]-1))
    # print('float:', df.shape[1], int(max_features*(df.shape[1]-1)))
  else :
    features_size = df.shape[1]-1
  #   print('not float:',df.shape[1]-1)
  # print(max_features)
  # print(features_size)

  # random features
  sample_df = df.drop(y,axis=1).sample(n=features_size, axis='columns')
  sample_df[y] = df[y]
  # print(sample_df)

  # select best split measure
  bsm = best_split_measure(sample_df, y)
  # print(bsm)
  node.name=bsm
  if df[bsm].dtype == 'O':
    for e in df[bsm]:
      split_df = df[df[bsm]==e]
      child_node = Node('child')
      node.add_child(e,child_node)
      RPA(child_node,split_df,y,max_depth-1, max_features)
  else:

    threshold = nume_best_split_value(df,bsm,y)
    # print('-----------------')
    # print(bsm,':',threshold)
    # print(threshold)
    # print(df[bsm].to_list())
    node.split_value = threshold
    left_df = df[df[bsm]<=threshold]
    right_df = df[df[bsm]>threshold]
    # if left_df or right_df is empty, return majority
    if len(left_df) == 0 or len(right_df) == 0:
      class_dict = df[y].value_counts().to_dict()
      node.name = max(class_dict, key=class_dict.get)
      node.split_value = None
      return
    # print('left df size:',len(left_df),',right df size:',len(right_df))
    else :
      left_node = Node('left')
      right_node = Node('right')
      node.add_child('left',left_node)
      node.add_child('right',right_node)
      RPA(left_node,left_df,y,max_depth-1, max_features)
      RPA(right_node,right_df,y,max_depth-1, max_features)
    
def train_test_split(data, class_col, min_class=None, maj_class=None, test_size=0.2):
    '''
    divide data into training set and testing set by maintain the ratio of minority instances and majority instances
    '''
    # define minor class and major class
    if min_class == None or maj_class == None:
      min_class, maj_class = minor_major_class(data, class_col)
    #divide data into minor data and major data
    min_data, maj_data = data[data[class_col] == min_class],data[data[class_col]==maj_class]

    #shuffle data
    min_data, maj_data = min_data.sample(frac=1), maj_data.sample(frac=1)
    min_thres, maj_thres = int(test_size*min_data.shape[0]), int(test_size*maj_data.shape[0]) # threshold for patition data

    #divide data
    maj_test, maj_train = maj_data[:maj_thres], maj_data[maj_thres:]
    min_test, min_train = min_data[:min_thres], min_data[min_thres:]
    d_train = pd.concat([maj_train, min_train], ignore_index = True)
    d_test = pd.concat([maj_test, min_test], ignore_index = True)
    # d_train = maj_train.append(min_train, ignore_index=True)
    # d_test = maj_test.append(min_test, ignore_index=True)
    return d_train, d_test

def predict(tree,df):
    '''
    predict a class of an individual data 'df' using decision tree 'tree'
    '''

    if tree.is_leaf():
      return tree.name
    cond = df[tree.name]
    if isinstance(cond,str):
      pred_class = predict(tree.children[cond],df)
    else :
      threshold = tree.split_value
      if (cond<threshold):
          pred_class = predict(tree.children['left'],df)
      else:
          pred_class = predict(tree.children['right'],df)

    return pred_class

def predict_all(node,df):
    '''
    predict a class of all data in 'df' using decision tree 'tree'
    '''

    pred = []
    for i in range(df.shape[0]):
      p = predict(node,df.iloc[i])
      pred.append(p)

    return pred

def confusion_matrix(output, label, min_c, maj_c):
    """
    Return confusion matrix from prediction result as dict

    Only for binary-class data set

    |-----------------|-----------------|
                        |  Actual value   |
                        |Positive|Negative|
    ------------------|--------|--------|
    Predicted|Positive|   TP   |   FP   |
                |--------|--------|--------|
        value  |Negative|   FN   |   TN   |
    ---------|--------|--------|--------|

    Parameter:
    - output : Result from predicting data
    - label : Actual class of data
    - min_c : minority class
    - maj_c : majority class

    """
    # result_dict = df[['Actual','Predicted']].value_counts().to_dict()
    result_df = pd.DataFrame({'Actual':label, 'Predicted':output})
    TP = result_df[(result_df.Actual == min_c) & (result_df.Predicted == min_c)].shape[0]
    FP = result_df[(result_df.Actual == maj_c) & (result_df.Predicted == min_c)].shape[0]
    FN = result_df[(result_df.Actual == min_c) & (result_df.Predicted == maj_c)].shape[0]
    TN = result_df[(result_df.Actual == maj_c) & (result_df.Predicted == maj_c)].shape[0]

    # row_name = ['Actual Positive','Actual Negative']
    # col_name = ['Predicted Positive','Predicted Negative']
    # min_c = minor_class(df,y)
    # maj_c = major_class(df,y)
    confusion_dict = {'TP':TP,'FP':FP,'FN':FN,'TN':TN}
    # confusion_df = pd.DataFrame(,index = row_name,columns=col_name)
    return confusion_dict

def Precision(conf_dict):
    """
    Return precision of the confusion matrix
    """
    if conf_dict['TP']+conf_dict['FP'] == 0:
      return 0
    prec = conf_dict['TP']/(conf_dict['TP']+conf_dict['FP'])
    return prec

def Recall(conf_dict):
    """
    Return recall of the confusion matrix
    """
    if conf_dict['TP']+conf_dict['FN'] == 0:
      return 0
    recall = conf_dict['TP']/(conf_dict['TP']+conf_dict['FN'])
    return recall

def Accuracy(conf_dict):
    """
    Return accuracy of the confusion matrix
    """
    if conf_dict['TP']+conf_dict['FP']+conf_dict['FN']+conf_dict['TN'] == 0:
      return 0
    acc = (conf_dict['TP']+conf_dict['TN'])/(conf_dict['TP']+conf_dict['FP']+conf_dict['FN']+conf_dict['TN'])
    return acc

def F1(conf_dict):
    """
    Return F1 score of the confusion matrix
    """
    prec = Precision(conf_dict)
    recall = Recall(conf_dict)
    if prec+recall == 0:
      return 0
    f_measure = 2*prec*recall/(prec+recall)
    return f_measure

def GM(conf_dict):
    """
    Return Geometric mean of the confusion matrix
    """
    prec = Precision(conf_dict)
    recall = Recall(conf_dict)

    GM = (prec*recall)**(1/2)
    return GM

def inner_fence_range(dat,col):
  '''
  calculate and return lower inner fence and upper inner fence. The formula of lower inner fence and upper inner fence are
    lower inner fence = q1-1.5*IQR
    upper inner fence = q3+1.5*IQR
    where 
    q1 = data at first quartile
    q3 = data at third quartile
    IQR = q3-q1 (Inter quartile range)
  '''
  q1 = dat[col].quantile(0.25)
  q3 = dat[col].quantile(0.75)
  IQR = q3-q1
  lower_inner_fence = q1-1.5*IQR
  upper_inner_fence = q3+1.5*IQR
  return [lower_inner_fence,upper_inner_fence]

def MCE(dat,col,y):
    '''
    Find subset of instances within the minority range and return best split value
    '''
    # counts = dat[y].value_counts().to_dict()
    # minor = min(counts, key=counts.get)
    minor, major = minor_major_class(dat,y)
    minor_df = dat[dat[y]==minor]
    # create set of minority instances without outlier
    l,u = inner_fence_range(minor_df,col)
    inner_minor_df = minor_df[(minor_df[col]>=l) & (minor_df[col]<=u)][col]
    # create the minority range
    lower_bound = inner_minor_df.min()
    upper_bound = inner_minor_df.max()
    # compute the subset of instances within the minority range
    minority_df = dat[(dat[col]>=lower_bound) & (dat[col]<=upper_bound)]
    # print("minor df")
    # print(minority_df)
    split_con = nume_best_split_value(minority_df,col,y)
    #entropy = nume_entropy(dat,col,y,split_con)
    return(split_con)

def MCE_best_split_measure(df,y):
    '''
    find best split measure of data 'df'
    '''
    attr_list = df.drop(y,axis=1).columns
    # assign initial best split attribute and entropy
    best_split_attr = attr_list[0]
    best_split_entropy = 0
    if df[best_split_attr].dtypes=='O':
      best_split_entropy = cate_entropy(df,best_split_attr,y)
    else :
      nume_split = nume_best_split_value(df,best_split_attr,y)
      best_split_entropy = nume_entropy(df,best_split_attr,y,nume_split)

    # find attribute with minimum entropy (MCE for numerical)
    for e in attr_list[1:]:
      if df[e].dtypes=='O':
          curr_entropy = cate_entropy(df,e,y)
      else :
          nume_split = MCE(df,e,y)
          curr_entropy = nume_entropy(df,e,y,nume_split)

      if curr_entropy < best_split_entropy:
          best_split_attr = e
          best_split_entropy = curr_entropy

    return best_split_attr

def MCDT(node,df,y,max_depth = 7, max_features = None):
    '''
    Build and return Minority condensation decision tree
    '''
    # return leaf node if there is only 1 distinct value in class column
    if stop_criteria(df,y) == True:
      c = df[y].unique()[0]
      node.name = c
      return

    # return leaf node if tree reach max depth
    if max_depth == 0:
      class_dict = df[y].value_counts().to_dict()
      node.name = max(class_dict, key=class_dict.get)
      return

    # drop column with 1 distinct value (except column of class)
    for col in df.drop(y,axis=1).columns:
      if len(df[col].unique()) == 1:
          df = df.drop(col,axis=1)

    # if there is no attribute left, then return leaf node with majority
    if len(df.columns)==1:
      class_dict = df[y].value_counts().to_dict()
      node.name = max(class_dict, key=class_dict.get)
      return

    # define number of features to consider
    if max_features == None:
      max_features = int(0.5*(len(df.columns)-1))
    elif isinstance(max_features, float):
      max_features = int(max_features*(len(df.columns)-1))

    # random features
    sample_df = df.drop(y,axis=1).sample(n=max_features, axis='columns')
    sample_df[y] = df[y]

    # select best split measure
    bsm = MCE_best_split_measure(sample_df, y)
    # print(bsm)
    node.name=bsm
    if df[bsm].dtypes == 'O':
      for e in df[bsm]:
          split_df = df[df[bsm]==e]
          child_node = Node('child')
          node.add_child(e,child_node)
          RPA(child_node,split_df,y,max_depth-1)
    else:

      threshold = nume_best_split_value(df,bsm,y)
      node.split_value = threshold
      left_df = df[df[bsm]<=threshold]
      right_df = df[df[bsm]>threshold]
      # if left_df or right_df is empty, return majority
      if len(left_df) == 0 or len(right_df) == 0:
          class_dict = df[y].value_counts().to_dict()
          node.name = max(class_dict, key=class_dict.get)
          node.split_value = None
          return
      else :
          left_node = Node('left')
          right_node = Node('right')
          node.add_child('left',left_node)
          node.add_child('right',right_node)
          RPA(left_node,left_df,y,max_depth-1)
          RPA(right_node,right_df,y,max_depth-1)