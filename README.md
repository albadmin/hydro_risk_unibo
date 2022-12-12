<a name="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This repository contains the source code material for the article entitled "Geomorhpic Flood Hazzard Mapping: from Floodplain Delination to Flood-Hazard Characterization", submitted for possible 
publication at the Environmental Modelling and Software Journal, Elsevier. Below is reported the abstract so to better contextualize our work.

 <div onmousedown="return false" onselectstart="return false">
<i>Recent literature reports on many applications of geomorphic indices retrieved from the analysis of Digital Elevation Models (DEMs) to flood hazard modelling and mapping. DEM-based techniques are generally trained on reference inundation or flood-hazard maps and are nowadays well-established and simpler alternatives to resource-intensive hydrodynamic models for flood hazard mapping. Our study highlights and addresses some limitations of the conventional application of such techniques, which to date are mostly targeting floodplain delineation, contributing to advancing our understanding of how to fully exploit their potential for computationally efficient and geographically consistent characterization of flood hazards across large geographical regions. We focus on three important aspects: (a) the accuracy, availability, and information content of input information (i.e., DEMs and reference flood-hazard maps); (b) how to optimize the efficiency of the computational pipeline and the integration of various software libraries available in the literature when such techniques are applied on large and very large DEMs; (c) how to best profit from the outcome of geomorphic flood hazard assessment. Our results (a) show the remarkable role played by input information; (b) exemplify the huge potential offered by computational pipeline optimization; (c) suggest that geomorphic flood hazard maps using continuous indices (e.g. inundation p-value from decision trees; raw geomorphic index; etc.) should always be preferred to binary flood-hazard map obtained by thresholding the continuous indices themselves (i.e. differentiating between likely and unlikely floodable pixels)</i>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The code is written in Python leveraging the <u>multiprocessing</u> and other libraries making use of thread/process-safe data structures. 
Parts of the pipeline (dem_analysis) rely on already available functions based on the <a href="https://github.com/dtarb/TauDEM">TauDEM</a> software suite.
<p>A direct link on how to install TauDEM and the MPI (Message Passing Interface) support can be consulted <a href="https://github.com/dtarb/TauDEM">here</a>.</p>

<!-- USAGE EXAMPLES -->
## Usage

The project structure is as follows:

- <b>io</b>: is the I/O directory where necessary program input/output is stored. The directory contains the MERIT DEM map and Hazzard map
- <b>P1-dem_analysis</b>.py: dem analysis phase, containing the macro-phases as described in the article.
- <b>P2.1-clf_calibration_buffer</b>.py: script containing the logic behind the buffer area calculation 
- <b>P2.2-clf_cross_validation</b>.py: script containing the cross validation source code
- <b>P2.3-clf_training</b>.py: scripts containing the training part 

<p>The scripts need to be run on sequence, starting from the DEM analysis used to generate the relevant geomorphic indexes. The later are then user by the rest of the processing 
pipeline used to train the machine learning algorithm.</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

FOr more information on the processing pipeline and algorithmic details, please contact:
- Andrea Magnini ()
- Armir Bujari ()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Project funded by XYZ
<p align="right">(<a href="#readme-top">back to top</a>)</p>


