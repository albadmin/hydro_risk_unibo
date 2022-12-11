#!/usr/bin/env python
# coding: utf-8

# TRAINING OF A DECISION TREE CLASSIFIER FOR CREATING A BINARY MAP OF HYDRAULIC RISK

# '''
# This script works as a computation sheet to fill with:
#     names of the input files and variables
#     names of the working directory
#     name of the shapefile of the study area
# 
# Output files are:
#     joblib file with the model to train
#     output binary map of hydraulic risk
# 
# '''

# '''
# Workflow:
#     opening input files
#     creating buffer calibration area
#     cross-validation (if not needed, change variable 'cv_comput')
#     training of the model (if cv is performed, best parameters will be taken from that,
#                            if not, they have to be set by the user)
#     validating the model with test set
#     using the model to create raster output map
#     
# 
# NB : input variables need to be changed by the user, in particular:
#     buffer_dist [m]
#     cv_comput
# 
# '''


#================================================================================================================
# (1) - INPUT FILES NAMES
#===============================================================================================================

import gc
from datetime import datetime
import os 
import sys
import numpy as np
from datetime import datetime
from osgeo import gdal
import rasterio
import fiona
import rasterio.mask
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from joblib import dump


# Modify next cell to change input/output filenames and code settings

print("Starting classification analysis")
print(datetime.now())

directory = r"/home/albadmin/Desktop/UNIBO/Progetti/Completed/Leitha-UniPOL/article_pipeline_revisit/io" #work directory, where all files reside and others are produced (stored). 
# Input files (.tif)
input_indexes = ['index_sd8.tif', 'MERIT_nordit_dem.tif',
                 'index_gfi.tif','index_hand.tif', 
                 'index_D.tif','index_lgfi.tif']
input_indexes_name =['slope', 'dem', 'gfi', 'hand', 'dist','lgfi']
target_map = 'hazmap_nordit_pgra_p1.tif' # Target map (.tif)

# Calibration area (.tif)
buffer_dist = 2000 ##radius of the buffer around flood-prone areas in target map
buffer_map = 'calibr_buffer.tif'
shape_area = 'Maschera_Po_estesa_3035.shp'

# Name to give the model to save it
model_name = ('dec_tree_clf.joblib')
# Name for the output binary map
output_map = 'output_dec_tree_clf.tif'

# Nodata for the input dataset and for the new file to create 
ML_nodata = -9999

# Parameteres for cross-validation
cv_comput = 1
max_depth_cv = [45, 60]
min_samples_leaf_cv = [500, 600]
# Parameters for tree training
max_depth = 45 #15
min_sample_leaf = 500 #60

#================================================================================================================
# (3) - USEFUL FUNCTIONS
#=================================================================================================================

print("Creating buffer raster file")
print(datetime.now())

os.chdir(directory)


#================================================================================================================
# (5) - OPENING FILES ( shuffling, changing nodata values, flattening )
#=================================================================================================================

print("Starting actual training: Step #2-CLF")
print("Opening files: shuffling, flattening...")
print(datetime.now())

seed = np.random.randint(0,10000)
factors_list = []

# Creating INPUT dataframe 
for factor in range(len(input_indexes)):
    print("-->Loading {} at {}".format(input_indexes[factor], datetime.now()))
    with rasterio.open(input_indexes[factor]) as src:
        factor_array = src.read()
        nodata = src.nodata
        factor_array2 = np.where(factor_array== nodata, ML_nodata, factor_array)
        factor_array2 = np.where((np.isnan(factor_array2)), ML_nodata, factor_array2)
        factor_array2 = np.ravel(factor_array2)
        np.random.seed(seed)
        np.random.shuffle(factor_array2)
        factors_list.append(factor_array2)
print("-->Loading target dataframe {}".format(datetime.now()))
# Creating TARGET dataframe
with rasterio.open(target_map) as src:
    meta = src.meta.copy()
    ref_haz_map_array = src.read()
    nodata = src.nodata
print("-->Loaded target dataframe {}".format(datetime.now()))

print("-->Loading hazmap {}".format(datetime.now()))
with rasterio.open(buffer_map) as src:
    buffer_array = src.read()
    nodata_buffer = src.nodata
    ref_haz_map_array2 = np.where(buffer_array!=nodata_buffer, 0, ref_haz_map_array)
    ref_haz_map_array2 = np.where(ref_haz_map_array!=nodata, ref_haz_map_array,
                                  ref_haz_map_array2)
    ref_haz_map_array2 = np.ravel(ref_haz_map_array2)
    np.random.seed(seed)
    np.random.shuffle(ref_haz_map_array2)
print("Loaded hazmap {}".format(datetime.now()))

#================================================================================================================
# (6) - INPUT DATA MANIPULATION
#=================================================================================================================

print("Input data manipulation")
print(datetime.now())

for factor in range(len(factors_list)):
   # Deleting pixels where target map is nodata from input indexes
   factors_list[factor] = np.delete(factors_list[factor],
                                    np.where(ref_haz_map_array2 == nodata))

# Deleting pixels where target map is nodata from target map
ref_haz_map_array2 = np.delete(ref_haz_map_array2, np.where(ref_haz_map_array2== nodata))
ref_haz_map_array2 = np.where(ref_haz_map_array2 == 0, nodata, ref_haz_map_array2)

# Deleting pixels where at least 1 input feature is nodata 
factors_list2 = factors_list.copy()
for j in range(len(factors_list)):
   factors_list2 = factors_list.copy()
   ref_haz_map_array2 = np.delete(ref_haz_map_array2, np.where(factors_list2[j] == ML_nodata))
   for i in range(len(factors_list)):
       factors_list[i] = np.delete(factors_list[i], np.where(factors_list2[j] == ML_nodata))
       
# Creating dataframe with input features preprocessed and randomized
df_array = np.column_stack(factors_list)


#================================================================================================================
# (7) - SPLITTING SOURCE DATASET INTO TRAINING AND TEST SET
#=================================================================================================================


print("Splitting source dataset")
print(datetime.now())


# split 85% of the dataset for training
print("Factor_array2 length is {} and perc {}".format(len(factor_array2), int(len(factor_array2)*15/100)))
df = pd.DataFrame(data=df_array.astype(np.float32), columns= input_indexes_name)
perc_to_split = int(len(df['gfi'])*15/100)
print("-->Length of original data frame df: {}".format(len(df.index)))
X = df [perc_to_split:]
print("-->Length of data frame X: {}".format(len(X.index)))
Y = ref_haz_map_array2 [perc_to_split:,]
print("-->Length of data frame Y: {}".format(len(Y)))
Y = Y.reshape(-1,1)
print("--->Length of data frame Y: {}".format(len(Y)))
# 15% of the dataset for test set
X2 = df[:perc_to_split] # test input features
Y2 = ref_haz_map_array2[:perc_to_split,] # test target
Y2 = Y2.reshape(-1,1)


#================================================================================================================
# (8) - CROSS-VALIDATION - To find best parameters for the training
#=================================================================================================================


print("Cross-validation")
print(datetime.now())

if cv_comput ==1:
    cv_mod_init = tree.DecisionTreeClassifier()
    def tss_func(y_true, y_pred):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        tss = (tp / (tp + fn)) + (tn /(tn + fp)) - 1
        return tss
    score = metrics.make_scorer(tss_func, greater_is_better=True)
            
    # Change variable 'params' for calibration!
    params = {'max_depth':max_depth_cv, 'min_samples_leaf': min_samples_leaf_cv}
    cv_mod = GridSearchCV(cv_mod_init, params, scoring = score, refit=True)
    cv_mod = cv_mod.fit(X, Y) 
    
    print('\n CROSS-VALIDATION WITH TRAINING SET ==========================================================================')
    print('Best parameters: ', cv_mod.best_params_)
    print('Best score: ', cv_mod.best_score_)
else: pass


#================================================================================================================
# (9) - MODEL TRAINING - Modify to set calibration parameters!!!
#=================================================================================================================


print("Training")
print(datetime.now())

# Hyperparameters of the tree
if cv_comput==1:
    best_params = cv_mod.best_params_
    max_depth = int(best_params['max_depth'])
    min_samples_leaf = int(best_params['min_samples_leaf'])
else: pass

# Training of the tree
mod = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
mod = mod.fit(X, Y) 

print('\n METRICS FOR THE TRAINING SET ==========================================================================')
print('Feature importance: ', mod.feature_importances_)
Y_pred = mod.predict(X)
tn, fp, fn, tp = metrics.confusion_matrix(Y, Y_pred).ravel()
tss = (tp /(tp + fn)) + (tn /(tn + fp)) - 1
print ('TSS for the model: ', tss)
print('Accuracy: ', metrics.accuracy_score(Y, Y_pred))
print('Precision: ', metrics.precision_score(Y, Y_pred))
print('Recall: ', metrics.recall_score(Y, Y_pred))
print('F1: ', metrics.f1_score(Y, Y_pred))


#================================================================================================================
# (7) - VALIDATION WITH TEST SET
#=================================================================================================================


print("Validation over test set")
print(datetime.now())

# Using the model to predict output for the test set
Y2_pred = mod.predict(X2)

# Computing metrics with test set
print('\n VALIDATION WITH TEST SET ==================================================================================')
tn, fp, fn, tp = metrics.confusion_matrix(Y2, Y2_pred).ravel()
tss = (tp /(tp + fn)) + (tn /(tn + fp)) - 1
print ('TSS for the model: ', tss)
print('Accuracy: ', metrics.accuracy_score(Y2, Y2_pred))
print('Precision: ', metrics.precision_score(Y2, Y2_pred))
print('Recall: ', metrics.recall_score(Y2, Y2_pred))
print('F1: ', metrics.f1_score(Y2, Y2_pred))

#================================================================================================================
# (8) - ML MODEL SAVING
#=================================================================================================================


try:
    dump(mod, directory + "/" + model_name)
except:
    pass



print("Ending CV analysis")
print(datetime.now())


