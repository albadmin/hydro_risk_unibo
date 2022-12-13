
#!/usr/bin/env python
# coding: utf-8

# """
# DEM ANALYSIS WITH D8 ALGORITHM
# 
# This script can be used to compute some geomoprphic features from an input DEM:
#     input: DEM of the study areaf
#     output: d8 (flow direction); ad8 (flow acccumulation); slope; 
#             D (distance from nearest stream); HAND; GFI; LGFI 
# 
# Computation of HAND, GFI and LGFI can be very long, so, if it's not required,
# put variable 'hand_comp' to 0
# 


# =================================================================================
# (1) LIBRARY
# =================================================================================


import os
import gc
import multiprocessing
from multiprocessing import Pool, RawArray, Manager
import sys
import subprocess
from datetime import datetime
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from pysheds.grid import Grid
import rasterio
import fiona
import rasterio.mask


# =================================================================================
# (2) DEFINITION OF INPUT FILES
# =================================================================================


# Modify next cell to change input/output filenames or code settings


# Pre existing input files
path = r'/home/albadmin/Desktop/UNIBO/Progetti/Completed/Leitha-UniPOL/article_pipeline_revisit/io' #working directory. The python script can be launched anywhere.
l_demvoid_name = "MERIT_nordit_dem.tif" #DEM map, currently set to "Nord Italy MERIT Dem"
watershed_shape ='Maschera_Po_estesa_3035.shp' #watershed shape file
thresh = 5000 #threshold parameter used by a TauDEM invocation. For mertiDEM type of maps the parameter is 5000. This is scaled (x16) for EUDEM.
basin = 'index' #prefix for the different computed index files.
ad8_calculation = True #variable denoting whether ad8 (TauDEM) index should be computed or not
D_calculation = True #variable denoting whether D (TauDEM) index should be computed or not
hand_comp = True #variable denoting whether HAND and related indexes (gfi, lfgi) should be computed or not

# suffic for the various index files.
l_fel_name = basin + '_fel.tif'
l_sd8_name = basin + '_sd8.tif'
l_d8_name = basin + '_d8.tif'
l_ad8_name = basin + '_ad8.tif'
l_stream = basin + '_stream.tif'
l_D = basin + '_D.tif'
filled_depressions_fname = basin + "_depressions.tif"

hand_name = basin + '_hand.tif'
gfi_name = basin + '_gfi.tif'
lgfi_name = basin + '_lgfi.tif'
auxiliary_hnet_matrix="h_aux.tif"

# =================================================================================
# (6) COMPUTATION OF HAND: opening needed files
# =================================================================================

print("Starting DEM analysis")
print(datetime.now().time())

os.chdir(path)


# =================================================================================
# (0) USEFUL FUNCTIONS
# =================================================================================


# Useful functions for the next steps:
def test(i,j):
    if not (0 <= i <= rows-1): # out of bounds N-S
        return 1
    if not (0 <= j <= cols-1): # out of bounds W-E
        return 1
    if d8[i,j] == -9999: # out of drainage divide
        return 1
    return 0

def step(i, j):
    # D8 flow direction coding + corresponding #cells increment/decrement
    di = {1:0, 8:1, 7:1, 6:1, 5:0, 4:-1, 3:-1, 2:-1} # N-S directions
    dj = {1:1, 8:1, 7:0, 6:-1, 5:-1, 4:-1, 3:0, 2:1} # W-E directions
    # new cell indices by increment/decrement
    xnw = i+di[d8[i,j]]
    ynw = j+dj[d8[i,j]]
    return xnw, ynw

def geotiff_clipping(input_tiff, output_tiff, shape_file, nodata):
    """
    input_tiff = path to input file
    output_tiff = 'output.tif'
    shape_file = path to input shape
    nodata = example: float('nan'), -9999, 0...
    """
    with fiona.open(shape_file, "r") as shapefile:
        features = [feature["geometry"] for feature in shapefile]
   
    with rasterio.open(input_tiff) as src:
        out_image, out_transform = rasterio.mask.mask(src, features,
                                                        crop=True)
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
    

# =================================================================================
# (1) COMPUTATION OF PITFILLING and FLAT RESOLVING 
# =================================================================================


command = ' '.join(['mpirun -n 8 pitremove',
                        '-z',
                        l_demvoid_name,
                        '-fel',
                        filled_depressions_fname])
print("--->Preping for 1st TauDEM invocation" + command)
process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
output = process.stdout
print(output)

with rasterio.open(filled_depressions_fname) as src:
    out_meta = src.meta.copy()
    array = src.read()
    nodata = src.nodata
    array2 = np.where(array==nodata, -9999, array)
    out_meta.update ({'nodata': -9999,
                      'crs': 'EPSG: 3035'})
    with rasterio.open (filled_depressions_fname, 'w', **out_meta) as dst:
        dst.write(array2)

# prep. grid for flat resolve
grid = Grid.from_raster(filled_depressions_fname, data_name = 'dcon')
dcon = grid.view('dcon', nodata = -9999).astype(np.float32)

# resolve flats in DEM
grid.resolve_flats('dcon', out_name='demcon',
                   nodata_in = -9999, nodata_out = -9999)
demcon = grid.view('demcon', nodata = -9999).astype(np.float32)

with rasterio.open(l_demvoid_name) as src:
    meta = src.meta.copy()
    with rasterio.open(l_fel_name, 'w', **meta) as dst:
        dst.write(demcon, 1)


# =================================================================================
# (2.1) COMPUTATION OF D8 FLOW DIRECTION AND ACCUMULATION 
# =================================================================================


if ad8_calculation:
    # Computing slope and flow direction
    command = ' '.join(['mpirun -n 8 d8flowdir',
                            '-fel',
                            l_fel_name,
                            '-p',
                            l_d8_name,
                            '-sd8',
                            l_sd8_name])
    print("--->Preping for 2nd TauDEM invocation" + command)
    print(datetime.now().time())
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    output = process.stdout
    print(output)
    print(datetime.now().time())
    # Computing flow accumulation 
    command = ' '.join(['mpirun -n 8 aread8',
                            '-p',
                            l_d8_name,
                            '-ad8',
                            l_ad8_name,
                            '-nc'])
    print("--->Preping for 3rd TauDEM invocation" + command)
    print(datetime.now().time())
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    output = process.stdout
    print(output)
    print(datetime.now().time())

# =================================================================================
# (2.2) POSTPROCESSING of produced files
# =================================================================================


if ad8_calculation:
# Post processing of outputs from TauDEM
    print("---->ad8_calculation")
    print(datetime.now().time())
    with rasterio.open(l_fel_name) as src:
        out_meta = src.meta.copy()
        array = src.read()
        nodata = src.nodata
        array2 = np.where(array==nodata, -9999, array)
        out_meta.update ({'nodata': -9999,
                          'crs': 'EPSG: 3035'})    
        with rasterio.open (l_fel_name, 'w', **out_meta) as dst:
            dst.write(array2)
    geotiff_clipping(l_fel_name,l_fel_name, watershed_shape,-9999)
        
    with rasterio.open(l_sd8_name) as src:
        out_meta = src.meta.copy()
        array = src.read()
        nodata = src.nodata
        array2 = np.where(array==nodata, -9999, array)
        out_meta.update ({'nodata': -9999,
                          'crs': 'EPSG: 3035'})      
        with rasterio.open (l_sd8_name, 'w', **out_meta) as dst:
            dst.write(array2)
    geotiff_clipping(l_sd8_name,l_sd8_name, watershed_shape,-9999)
    
    with rasterio.open(l_d8_name) as src:
        out_meta = src.meta.copy()
        array = src.read()
        nodata = src.nodata
        array2 = np.where(array==nodata, -9999, array)
        out_meta.update ({'nodata': -9999,
                          'crs': 'EPSG: 3035'})      
        with rasterio.open (l_d8_name, 'w', **out_meta) as dst:
            dst.write(array2)
    geotiff_clipping(l_d8_name,l_d8_name, watershed_shape,-9999)
      
    with rasterio.open(l_ad8_name) as src:
        out_meta = src.meta.copy()
        array = src.read()
        nodata = src.nodata
        array2 = np.where(array==nodata, -9999, array)
        out_meta.update ({'nodata': -9999,
                          'crs': 'EPSG: 3035'})      
        with rasterio.open (l_ad8_name, 'w', **out_meta) as dst:
            dst.write(array2)
    geotiff_clipping(l_fel_name,l_fel_name, watershed_shape,-9999)

# =================================================================================
# (2.3) COMPUTE DISTANCE FROM RIVER 
# =================================================================================

if D_calculation:
    print("---->D_calculation")
    print(datetime.now().time())
    # Definition of STREAM_CHANNEL with application of a threshold to file ad8
    with rasterio.open(l_ad8_name) as src:
        meta = src.meta.copy()
        array = src.read()
        array_stream = np.where(array>= thresh, array, -9999)
        with rasterio.open(l_stream, 'w', **meta) as dst:
            dst.write(array_stream)
        
    # Calcolo della D tramite TauDEM
    command = ' '.join(['mpirun -n 8 d8hdisttostrm',
                            '-p',
                            l_d8_name,
                            '-src',
                            l_stream,
                            '-dist',
                            l_D,
                            '-thresh',
                            str(thresh)])
    print("--->Preping for 4th TauDEM invocation" + command)
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    print(datetime.now().time())

    with rasterio.open(l_D) as src:
            out_meta = src.meta.copy()
            array = src.read()
            nodata = src.nodata
            array2 = np.where(array==nodata, -9999, array)
            out_meta.update ({'nodata': -9999,
                              'crs': 'EPSG: 3035'})      
            with rasterio.open (l_D, 'w', **out_meta) as dst:
                dst.write(array2)
    geotiff_clipping(l_D, l_D, watershed_shape,-9999)

# =================================================================================
# (2.4) COMPUTATION OF HAND: opening needed files
# =================================================================================

if hand_comp: 
    print("----> hand_comp")
    print(datetime.now().time())
    # Opening l_fel
    with rasterio.open(l_fel_name) as src:
        meta = src.meta.copy()
    grid = Grid.from_raster(l_fel_name, data_name='demcon')
    demcon = grid.view('demcon', nodata = -9999).astype(np.float32)
    # cell width times height
    cell_area = np.abs(grid.affine.a * grid.affine.e)
    rows, cols = grid.shape
    # l_sd8
    grid = Grid.from_raster(l_sd8_name, data_name='sd8')
    sd8 = grid.view('sd8', nodata = -9999).astype(np.float32)
    sd8[demcon == -9999] = -9999
    # l_d8
    grid = Grid.from_raster(l_d8_name, data_name = 'd8')
    d8 = grid.view('d8', nodata = -9999).astype(np.float32)
    # l_ad8 
    grid = Grid.from_raster(l_ad8_name, data_name = 'ad8')
    ad8 = grid.view('ad8', nodata = -9999).astype(np.float32)
    print("---> hand_comp finished opening files")
    print(datetime.now().time())

# =================================================================================
# (2) COMPUTE RIVER NETWORK with algorithm from Giannoni et al., 2005
# =================================================================================

# Minor part of the algorithm has been intentionally left out, due to IPR compliance
# clauses. This is the part for memorize already visited pixels along flow 
# direction, as described in Section 6.1 of the manuscript "MORPHIC FLOOD HAZARD 
# MAPPING: FROM FLOODPLAIN DELINEATION TO FLOOD-HAZARD CHARACTERIZATION"


if hand_comp: 
    print("---> hand_comp preparation of parameter computation")
    print(datetime.now().time())
    # Parameters for the computation
    k = 1.7
    th_channel = 1*10^5
    
    # Origin points for the river network
    A = np.multiply(ad8, cell_area)
    print("Parameter first multiplication")
    print(datetime.now().time())
    Sk = np.power(np.abs(sd8), k, dtype=np.float128)
    print("Parameter power operation")
    print(datetime.now().time())
    ASk = np.multiply(ad8, Sk, dtype=np.float128)
    print("Parameter second multiplication")
    print(datetime.now().time())
    # Binary mask with origin points
    init = np.zeros(np.shape(d8), dtype = np.float32)
    init[ASk > th_channel] = 1
    init[demcon == -9999] = -9999
    print("Parameter init operations")
    print(datetime.now().time())
    # Indexes of the orgin points
    x0,y0 = np.where(ASk > th_channel) 


    rnet = init.copy()
    # A global dictionary storing the variables passed from the initializer.
    var_dict = {}

    def init_worker(X, X_shape):
        # Using a dictionary is not strictly necessary. You can also
        # use global variables.
        var_dict['X'] = X
        var_dict['X_shape'] = X_shape

    #in-memory, shared, lock-free, data structure. Essential to guarantee correctness is to partition it with each process acting on
    #independent set of indexes. If cross-boundary conditions ocurr, announce them.
    X_shape = rnet.shape
    X = RawArray(np.ctypeslib.as_ctypes_type(np.float32), X_shape[0] * X_shape[1]) # 'd'
    X_np = np.frombuffer(X, np.ctypeslib.as_ctypes_type(np.float32)).reshape(X_shape) #ab np.float64
    np.copyto(X_np, rnet)
    

    #run configuration: 8 processes, 4 is the axis split factor. 
    run_conf = [(8,4,2)]
    nr_partitions, nr_partitions_x, nr_partitions_y = run_conf[0]

    #inter-process-com (IPC) handler. Generates python-called pickleable obj which can be accessed remotely through a distributed (C-S) locking mecchanism.
    manager = multiprocessing.Manager()
    partition_table = {}
    points_partition = {}

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

    #first call to relax points, acting upon a memory-shared, lock-free, data structure
    def firstPass(ptable, partition_bounds, points_partition, queue_table):
        shared = np.frombuffer(var_dict['X'], np.ctypeslib.as_ctypes_type(np.float32)).reshape(var_dict['X_shape']) #ab np.float64

        for a,b in points_partition:
            x = a; y = b
            while True:
                if test(x,y) == 1:
                    break
                x,y = step(x,y)
                #enforce check not to cross border, avoiding any race-condition(s).
                if isInPartition(partition_bounds,x,y):
                    shared[x,y] = 1
                else:
                    #got out of boundaries, deal with everything later on. This point is set to 1 in the begining, then percolation starts as usual (nextPass)
                    queue_table[getPartitionKey(ptable, x,y)].put((x,y))
                    break; 

    #called after the first step is performed, acting upon the same memory-shared, lock-free (efficient), data structure
    #the diff with the above is essentially the "first assignement" - could merge with firstPass and differentiate upon a parameter value.
    def nextPass(ptable, partition_bounds, points_partition, queue_table):
        shared = np.frombuffer(var_dict['X'], np.ctypeslib.as_ctypes_type(np.float32)).reshape(var_dict['X_shape'])
        for a,b in points_partition:
            x = a; y = b
            shared[x,y] = 1
            while True:
                if test(x,y) == 1:
                    break
                x,y = step(x,y)
                if not isInPartition(partition_bounds,x,y):
                    queue_table[getPartitionKey(ptable, x,y)].put((x,y)) #got out of own boundaries, com. to other
                    break
                shared[x,y] = 1

    #partitions the scenario (matrix) into non-overlapping equally-sized rectangulars over the axeses.
    #points tend to converge to certain areas and a more sensible partition strategy should be aware of this to achieve a good degree of ||.
    def fillInPartitionTable(shape, factor1, factor2):
        global partition_table
        x_axis = np.array_split(range(shape[0]), factor1)
        y_axis = np.array_split(range(shape[1]), factor2)
        pid = 0
        for a in x_axis:
            for b in y_axis:
                x = np.array((a[0], a[len(a)-1]))
                y = np.array((b[0], b[len(b)-1]))
                partition_table[pid] = (x,y)
                pid+=1
    
    #given a partition table, partition the next to visit candidate points.
    def partitionPoints(ptable, x0, y0):    
        global points_partition
        for a,b in zip(x0, y0):
            for key in ptable:
                if isInPartition(ptable[key], a, b):
                    points_partition[key].append((a,b))   

    firstTime = True         
    finished = False
    #Parallel algorithm computing the river network (computationally intensive task, hence parallelized)           
    print("Starting passes {}".format(datetime.now().time()))  
    with Pool(processes=nr_partitions, initializer=init_worker, initargs=(X, X_shape)) as pool:
        
        while not finished:
            print("Initializing partitions step #1")
            partition_table.clear()
            max_partitions, nr_partitions_x, nr_partitions_y = run_conf[0]
            fillInPartitionTable(rnet.shape, nr_partitions_x, nr_partitions_y)
            print("Initializing partitions step #2")    
            #create a list of shared manager queues: each partition has its own worker from pool, each having its own queue
            #queues are blocking and process safe (essential to correctness).
            queue_table = list(range(max_partitions))
            for key in partition_table:
                queue_table[key] = manager.Queue()
            #should clear partition points
            points_partition.clear()
            for i in range(max_partitions):
                points_partition[i] = []
            partitionPoints(partition_table, x0, y0)
            print("Running configuration {} with points {} at time {}".format(run_conf[0], len(x0), datetime.now().time()))
            if firstTime:
                #synchronous pool of processes.
                pool.starmap(firstPass, [(partition_table, partition_table[key], points_partition[key], queue_table) for key in partition_table])
                firstTime = False
            else:
                #synchronous pool of processes.
                pool.starmap(nextPass, [(partition_table, partition_table[key], points_partition[key], queue_table) for key in partition_table])            
            xAxis = []
            yAxis = []            
            #gather all the "boundary" points from the queues, fillin the next candidate x0, y0
            for key in partition_table:
                while not queue_table[key].empty():
                    a,b = queue_table[key].get()
                    xAxis.append(a)
                    yAxis.append(b)
            if len(xAxis) == 0:
                finished = True 
                break
            x0 = np.asarray(xAxis)
            y0 = np.asarray(yAxis)            

    pool.close()
    pool.join()

    rnet = X_np.copy()
    
    print("---> hand_comp: finished computing parameters")
    print(datetime.now().time())


# =================================================================================
# (8) COMPUTATION OF HAND
# =================================================================================

if hand_comp: 
    print("---> hand_comp: loading hand index")
    print(datetime.now().time())

    # Algorithm was intentionally left out, due to IPR compliance clauses; 
    # loading a precomputed HAND index. 
    # For information on the computation please refer to Section 2.1 of the 
    # manuscript "MORPHIC FLOOD HAZARD MAPPING: FROM FLOODPLAIN DELINEATION 
    # TO FLOOD-HAZARD CHARACTERIZATION"
    with rasterio.open(hand_name) as src:
        H = src.read(1)
        profile = src.profile    
    
# =================================================================================
# (9) COMPUTATION OF h, EXPECTED WATER DEPTH 
# =================================================================================

if hand_comp: 
    
    print("loading h/hnet auxiliary matrix")
    
    # Algorithm was intentionally left out, due to IPR compliance clauses; 
    # loading a precomputed water depth h. 
    # For information on the computation please refer to Section 2.1 of the 
    # manuscript "MORPHIC FLOOD HAZARD MAPPING: FROM FLOODPLAIN DELINEATION 
    # TO FLOOD-HAZARD CHARACTERIZATION"    with rasterio.open(auxiliary_hnet_matrix) as src:
        h = src.read(1)
        profile = src.profile    

# =================================================================================
# (10) COMPUTATION OF GFI
# =================================================================================

if hand_comp: 
    gfi = np.log(np.divide(h, H))
    
    with rasterio.open(gfi_name, 'w', **meta) as src:
        src.write(gfi, 1)

# =================================================================================
# (11) COMPUTATION OF LGFI
# =================================================================================

if hand_comp: 
    # Opening needed input files
    with rasterio.open(l_ad8_name) as src:
        h_array = src.read()
        profile = src.profile
        
    H_array = np.where(H<=0, 0.00001, H)
    H_array = np.where(H == -9999, -9999, H_array)
    h2_array = np.multiply(h_array, cell_area)  
    h2_array = np.multiply((np.float_power(h2_array, 0.3)),0.001)  
    
    
    H_array = np.ma.masked_invalid(H_array)
    h2_array = np.ma.masked_invalid(h2_array)
    
    lgfi = np.log(np.divide(h2_array, H_array))
    lgfi = lgfi.filled(-9999)
    
    with rasterio.open(lgfi_name,'w', **meta) as dst:
        dst.write(lgfi.astype(np.float32))

print("Script compute ended")
print(datetime.now().time())
