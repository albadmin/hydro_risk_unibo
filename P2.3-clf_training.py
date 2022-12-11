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
from joblib import dump, load


# Modify next cell to change input/output filenames and code settings

print("Starting classification analysis")
print(datetime.now())

# Directory where all the files are placed: input, target, training area
directory = r"/home/albadmin/Desktop/UNIBO/Progetti/Completed/Leitha-UniPOL/article_pipeline_revisit/io"
# Input files (.tif)
input_indexes = ['index_sd8.tif', 'MERIT_nordit_dem.tif',
                 'index_gfi.tif','index_hand.tif', 
                 'index_D.tif','index_lgfi.tif']
input_indexes_name =['slope', 'dem', 'gfi', 'hand', 'dist','lgfi']
# Target map (.tif)
target_map = 'hazmap_nordit_pgra_p1.tif' # Target map (.tif)
# Calibration area (.tif)
buffer_dist = 2000 #2000  #radius of the buffer around flood-prone areas in target map
buffer_map = 'calibr_buffer.tif'

# Name to give the model to save it
model_name = ('dec_tree_clf-remastered.joblib')
# Name for the output binary map
output_map = 'output_dec_tree_clf.tif'

# Nodata for the input dataset and for the new file to create 
ML_nodata = -9999


print("Creating buffer raster file")
print(datetime.now())

os.chdir(directory)


#================================================================================================================
# (1) - OPENING FILES ( shuffling, changing nodata values, flattening )
#=================================================================================================================

print("Starting actual training: Step #2-CLF")
print("Opening files: shuffling, flattening...")
print(datetime.now())

seed = np.random.randint(0,10000)

#load saved model.
mod = load(model_name)


#================================================================================================================
# (2) - USING THE MODEL TO PREDICT OUTPUT FOR THE ENTIRE STUDY AREA, to obtain output raster map
#=================================================================================================================


# Creating raster file for visualisation with GIS
factors_list_vis = []
for factor in range(len(input_indexes)):
    with rasterio.open(input_indexes[factor]) as src:
        print("Loading index {}".format(input_indexes[factor]))
        factor_array_vis = src.read()
        nodata = src.nodata
        factor_array_vis2 = np.where(factor_array_vis== nodata, ML_nodata,
                                     factor_array_vis)
        factor_array_vis2 = np.where((np.isnan(factor_array_vis2)), ML_nodata,
                                     factor_array_vis2)
        factor_array_vis2 = np.ravel(factor_array_vis2)
        factors_list_vis.append(factor_array_vis2)
df_array_vis = np.column_stack(factors_list_vis)
df_vis = pd.DataFrame(data=df_array_vis.astype(np.float32),
                      columns= input_indexes_name)
X_vis = df_array_vis
print("Starting prediction")
output = mod.predict(X_vis) 

# Postprocessig: creating array with shape of the study area
print("Postprocessing: creating array with shape of the study area")
try:
    study_area = df_vis['dem']
    study_area = study_area.to_numpy()
    nodata_study_area = ML_nodata
except:
    study_area = df_vis['gfi']
    study_area = study_area.to_numpy()
    nodata_study_area = ML_nodata

# Postprocessing of output: putting nodata outside from study area 
print("Output postprocessing")
output2 = np.where(study_area == nodata_study_area, nodata, output)

print("Saving tiff.")
with rasterio.open(target_map) as src:
    meta = src.meta.copy()
# Creating a .tif file with output binary map
output2 = np.reshape(output2, factor_array_vis.shape)
with rasterio.open( output_map, 'w', **meta) as dst:
    dst.write(output2.astype(np.float32))

#================================================================================================================
# (3) - COMPUTING EVALUATION METRICS FOR THE FINAL PREDICTION (output raster map)
#=================================================================================================================


# Calculating accuracy: just in TRAINING AREA
with rasterio.open(target_map) as src:
    ref_haz_map_array_vis = src.read()
    ref_haz_map_array_vis = np.ravel(ref_haz_map_array_vis)
    nodata = src.nodata
with rasterio.open(buffer_map) as src:
    buffer_array = src.read()
    nodata_buffer = src.nodata
    buffer_array = np.ravel(buffer_array)

output_vis2 = np.delete(output, np.where(buffer_array==nodata_buffer))        
ref_haz_map_array_vis2 = np.delete(ref_haz_map_array_vis,
                                   np.where(buffer_array==nodata_buffer))

# Calculating accuracy: in the WHOLE STUDY AREA
print('\n ACCURACY for the model applied to the whole area')

output_vis3 = np.delete(output, np.where(study_area==nodata_study_area))
ref_haz_map_array_vis3 = np.delete(ref_haz_map_array_vis,
                                   np.where(study_area==nodata_study_area))

tn, fp, fn, tp = metrics.confusion_matrix(ref_haz_map_array_vis3, output_vis3).ravel()
tss = (tp / (tp + fn)) + (tn /(tn + fp)) - 1

print ('TSS: ', tss)
print ('Accuracy: ',
       metrics.accuracy_score(ref_haz_map_array_vis3, output_vis3))
print('Precision: ',
      metrics.precision_score(ref_haz_map_array_vis3, output_vis3))
print('Recall: ',
      metrics.recall_score(ref_haz_map_array_vis3, output_vis3))
print('F1: ',
      metrics.f1_score(ref_haz_map_array_vis3, output_vis3))
 

print("Ending classification analysis")
print(datetime.now())
