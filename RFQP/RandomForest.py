import Bootstrapping as bt
import DecisionTree as dt
import pandas as pd
import numpy as np
import time

def RF(data, class_col, tree_no, s_size, max_depth = 7, max_features = 0.5):
    Tree_list = []
    # do bootstrapping
    Bootstrap_list = bt.bootstrap(data, s_size, tree_no)
    for d_i in Bootstrap_list:
        tree = dt.Node('root')
        dt.RPA(tree,d_i,class_col, max_depth, max_features)
        Tree_list.append(tree)
        time.sleep(1)
    return Tree_list

def RFQP(data, class_col, tree_no, s_size, npartition=4, normalize_method=None, max_depth=7, max_features=0.5):
  Tree_list = []

  # do bootstrapping
  Bootstrap_list = bt.QPBT(data, class_col, s_size, tree_no, npartition, normalize_method)

  for d_i in Bootstrap_list:

    # Build decision tree
    tree1 = dt.Node('root')
    dt.RPA(tree1,d_i,class_col, max_depth, max_features)
    Tree_list.append(tree1)

    time.sleep(1)

    # Build MCDT
    tree2 = dt.Node('root')
    dt.MCDT(tree2,d_i,class_col, max_depth, max_features)
    Tree_list.append(tree2)

  return Tree_list

def RF_predict(RF, data):
  '''
  Predict and return class of an individual data 'data' using random forest 'RF'
  '''
  result = []
  for tree in RF:
    result.append(dt.predict(tree,data))

  return max(result,key = result.count)

def RF_predict_all(RF, data):
    '''
    Predict and return class of all instances in 'data' using random forest 'RF'
    '''
    result = []
    for i in range(len(data)):
        result.append(RF_predict(RF,data.iloc[i]))

    return result