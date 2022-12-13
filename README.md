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

<i>Recent studies show how geomorphic descriptors, retrieved from digital elevation models (DEMs), can be used for flood hazard mapping. As strictly depending on the accuracy of the input DEMs and target flood hazard maps used for training, DEM-based flood hazard models can suffer from severe inconsistencies. Also, although these models are naturally suitable for large-scale ap-plications, we lack optimized computational tools for effectively handling big datasets. Our study specifically addresses these issues. On the one hand, we exemplify the huge advantages associated with optimization and parallelization of computational pipelines that integrate existing open-source tools and libraries for DEM-based flood hazard modelling; on the other hand, we show how to limit inconsistencies of DEM-based flood hazard models by (a) referring simultaneously to mul-tiple geomorphic indices, and (b) preferring a continuous representation of hazard to binary flood maps.</i>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The code is written in Python leveraging the <u>multiprocessing</u> and other libraries making use of thread/process-safe data structures. 
Parts of the pipeline (dem_analysis) rely on already available functions based on the <a href="https://github.com/dtarb/TauDEM">TauDEM</a> software suite.
<p>A direct link on how to install TauDEM and the MPI (Message Passing Interface) support can be consulted <a href="https://github.com/dtarb/TauDEM">here</a>.</p>

<!-- USAGE EXAMPLES -->
## Usage

The project structure is as follows:

- <b>io</b>: is the I/O directory where necessary program input/output is stored. The directory contains the MERIT DEM map, to be used to compute relevant input descriptors, and the Hazard map, to be used as target for the model training.
- <b>P1-dem_analysis</b>.py: DEM analysis phase, containing the computation of the geomorphic descriptors as described in the article.
- <b>P2.1-clf_calibration_buffer</b>.py: script containing the logic behind the buffer area calculation.
- <b>P2.2-clf_cross_validation</b>.py: script containing the cross validation source code.
- <b>P2.3-clf_training</b>.py: scripts containing the training part.

<p>The scripts need to be run on sequence, starting from the DEM analysis used to generate the relevant geomorphic indexes. The later are then user by the rest of the processing 
pipeline used to train the machine learning algorithm.</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

For more information on the processing pipeline and algorithmic details, please contact:
- Andrea Magnini (andrea.magnini@unibo.it)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Project funded by Leithà S.r.l., Unipol Group 
<p align="right">(<a href="#readme-top">back to top</a>)</p>


