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
import multiprocessing
from multiprocessing import Pool, RawArray, Manager

# Modify next cell to change input/output filenames and code settings

print("Starting preparation of calibr buffer: Step #1-CLF")
print(datetime.now())

# Directory where all the files are placed: input, target, training area
directory = r"/home/albadmin/Desktop/UNIBO/Progetti/Completed/Leitha-UniPOL/article_pipeline_revisit/io"

# Target map (.tif)
target_map = 'hazmap_nordit_pgra_p1.tif'
# Calibration area (.tif)
buffer_dist = 2000 #2000  #radius of the buffer around flood-prone areas in target map
buffer_map = 'calibr_buffer.tif'
shape_area = 'Maschera_Po_estesa_3035.shp'

# Nodata for the input dataset and for the new file to create 
ML_nodata = -9999


#================================================================================================================
# (2) - LIBRARIES
#=================================================================================================================

import os 
import sys
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, RawArray
from datetime import datetime
from osgeo import gdal
import rasterio
import fiona
import rasterio.mask


#================================================================================================================
# (1) - CLIPPING FUNCTION
#=================================================================================================================

def geotiff_clipping(input_tiff, output_tiff, shape_file, nodata):
    """
    Function to clip .tif files with a shapefile
    input_tiff = path to input file
    output_tiff = 'output.tif'
    shape_file = path to input shape
    nodata = example: float('nan'), -9999, 0...
    """
    with fiona.open(shape_file, "r") as shapefile:
        features = [feature["geometry"] for feature in shapefile]
    with rasterio.open(input_tiff) as src:
        out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform,
                 "nodata" : nodata
                 })
    with rasterio.open(output_tiff, "w", **out_meta) as dest:
        dest.write(out_image)
    return
    
print("Creating buffer raster file")
print(datetime.now())

os.chdir(directory)

d = gdal.Open(target_map)
geotrans=d.GetGeoTransform()
row=d.RasterYSize
col=d.RasterXSize
inband=d.GetRasterBand(1)
array = inband.ReadAsArray(0,0,col,row).astype(int)
Xcell_size=int(abs(geotrans[1]))
Ycell_size=int(abs(geotrans[5]))
cell_size = (Xcell_size+Ycell_size)/2
cell_dist=buffer_dist/cell_size
cell_dist=round(cell_dist)
shape = array.shape
marg=np.where(array==-9999, 0, array)

#shared data structure
X_shape = shape
X = RawArray('i', X_shape[0] * X_shape[1])
X_np = np.frombuffer(X, dtype=np.int32).reshape(X_shape)
# Copy data to our shared array.
np.copyto(X_np, marg)


#given a point (x,y) find the partition_id.
def getPartitionKey(ptable, x, y):    
    for key in ptable:
        if isInPartition(ptable[key], x, y):
            return key
    return -1

#
def firstPassBufferer(ptable, pbounds, qtable, val):
    #retrieve shared mem. structure.
    marg = np.frombuffer(var_dict['X'], dtype=np.int32).reshape(var_dict['X_shape'])
    for i in range(pbounds[0][0], pbounds[0][1]+1):
        for j in range(pbounds[1][0], pbounds[1][1]+1):
            if marg[i,j]==val:
                if isInPartition(pbounds, i+1,j):
                    if marg[i+1,j]==0: marg[i+1,j]= val+1
                else:
                    qtable[getPartitionKey(ptable, i+1,j)].put((i+1,j))
                if isInPartition(pbounds, i,j+1):
                    if marg[i,j+1]==0: marg[i,j+1]= val+1
                else:
                    qtable[getPartitionKey(ptable, i,j+1)].put((i,j+1))
                if isInPartition(pbounds, i+1,j+1):
                    if marg[i+1,j+1]==0: marg[i+1,j+1]= val+1
                else:
                    qtable[getPartitionKey(ptable, i+1,j+1)].put((i+1,j+1))
                if isInPartition(pbounds, i-1,j):
                    if marg[i-1,j]==0: marg[i-1,j]= val+1
                else:
                    qtable[getPartitionKey(ptable, i-1,j)].put((i-1,j))
                if isInPartition(pbounds, i,j-1):                    
                    if marg[i,j-1]==0: marg[i,j-1]= val+1
                else:
                    qtable[getPartitionKey(ptable, i,j-1)].put((i,j-1))
                if isInPartition(pbounds, i-1,j-1):                    
                    if marg[i-1,j-1]==0: marg[i-1,j-1]= val+1
                else:
                    qtable[getPartitionKey(ptable, i-1,j-1)].put((i-1,j-1))
                if isInPartition(pbounds, i+1,j-1):                    
                    if marg[i+1,j-1]==0: marg[i+1,j-1]= val+1
                else:
                    qtable[getPartitionKey(ptable, i+1,j-1)].put((i+1,j-1))
                if isInPartition(pbounds, i-1,j+1):
                    if marg[i-1,j+1]==0: marg[i-1,j+1]= val+1
                else: 
                    qtable[getPartitionKey(ptable, i-1,j+1)].put((i-1,j+1))
                

#patern is it waits until all have finished work.                
def nextPassBufferer(x0, y0, val):
    #get all shared data structures
    for a,b in zip(x0,y0):
        if X_np[a,b]==0: X_np[a,b]= val+1
                

# A global dictionary storing the variables passed from the initializer.
var_dict = {}
run_conf = [(12,4,3)]
nr_partitions, nr_partitions_x, nr_partitions_y = run_conf[0]

#inter-process-com (IPC) handler. Generates python-called pickleable obj which can be accessed remotely through a distributed (C-S) locking mecchanism.
manager = multiprocessing.Manager()
partition_table = {}
nextPassX0 = []
nextPassY0 = []
    
def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

#test whether point falls within this partition bounds. Process-safe (read only)
def isInPartition(partition_bounds, x, y):
    x_range = len(partition_bounds[0])
    y_range = len(partition_bounds[1])
    x_min = partition_bounds[0][0] #if len(x_range) > 0 else -1
    x_max = partition_bounds[0][1] #if len(x_range) == 2 elif len(x_range) == 1 partition_bounds[0][0] else -1
    if x_min <= x <= x_max:
        y_min = partition_bounds[1][0] #if len(y_range) > 0 else -1
        y_max = partition_bounds[1][1] #if len(y_range) == 2 elif len(y_range) == 1 partition_bounds[1][0] else -1 
        if y_min <= y <= y_max: 
            return True
    return False
        
#given a point (x,y) find the partition_id.
def getPartitionKey(ptable, x, y):    
    for key in ptable:
        if isInPartition(ptable[key], x, y):
            return key
    return -1
        
def fillInPartitionTable(shape, factor1, factor2):
    global partition_table
    x_axis = np.array_split(range(shape[0]-1), factor1)
    y_axis = np.array_split(range(shape[1]-1), factor2)
    pid = 0
    for a in x_axis:
       for b in y_axis:
           x = np.array((a[0], a[len(a)-1]))
           y = np.array((b[0], b[len(b)-1]))
           partition_table[pid] = (x,y)
           pid+=1
                
                
#main function
def splitter(shape, val):
    firstTime = True         
    finished = False           
    with Pool(processes=nr_partitions, initializer=init_worker, initargs=(X, X_shape)) as pool:
        while not finished:
            #ab: the above ops. (partition) can be done only once.
            partition_table.clear()
            max_partitions, nr_partitions_x, nr_partitions_y = run_conf[0]
            fillInPartitionTable(shape, nr_partitions_x, nr_partitions_y)
            #create a list of shared manager queues: each partition has its own worker from pool, each having its own queue
            #queues are blocking and process safe (essential to correctness).
            queue_table = list(range(max_partitions))
            for key in partition_table:
                queue_table[key] = manager.Queue()
            print("-->Running configuration {} at time {}".format(run_conf[0], datetime.now().time()))
            if firstTime:
                print("-->Performing first pass at time {}".format(datetime.now().time()))
                #synchronous pool of processes.
                pool.starmap(firstPassBufferer, [(partition_table, partition_table[key], queue_table, val) for key in partition_table])
                firstTime = False
            else:
                print("-->Performing second pass, correcting {} pass at time {}".format(len(nextPassY0), datetime.now().time()))            
                #synchronous pool of processes.
                nextPassBufferer(nextPassX0, nextPassY0, val)            
            #gather all the "boundary" points from the queues, fillin the next candidate x0, y0
            nextPassX0 = []
            nextPassY0 = []
            for key in partition_table:
                while not queue_table[key].empty():
                    a,b = queue_table[key].get()
                    nextPassX0.append(a)
                    nextPassY0.append(b)
            if len(nextPassX0) == 0:
                finished = True
    pool.close()
    pool.join()




#================================================================================================================
# (4) - CREATING BUFFER RASTER FILE ( shuffling, changing nodata values, flattening )
#=================================================================================================================

for i in range(cell_dist):
    print("Launching iteration {} of {} at {}".format(i, len(range(cell_dist)), datetime.now()))
    val = i+1
    splitter(shape, val)

marg = np.where(X_np==0, -9999, marg)
marg = np.where(marg!=-9999, 1, marg)    # trasformo la matrice in binaria (1, -9999)
with rasterio.open(target_map) as src:
    meta = src.meta.copy()
with rasterio.open(buffer_map,'w',**meta) as dst:
    dst.write(marg.astype(np.float32),1)

geotiff_clipping(buffer_map, buffer_map, shape_area, ML_nodata)
