## Course overview

In this hands-on course, students will work in groups to develop a comprehensive project that applies the AI tools and techniques learned throughout the master to a real-world problem. The project involves solving a complex challenge using a provided raw dataset, and students will design and implement a solution from scratch.

Key Project Components:

**Repository Development**: Students will create a repository to document their code, data structures, and workflow, ensuring transparency and reproducibility.
**Python Programming**: Building on previous coursework, students will write Python code from scratch to develop their solution.
Machine Learning Model Development: Students will design and implement machine learning models tailored to their specific problem, processing raw datasets and generating metadata as needed.
**Data Structure Interface**: Students will create an interface for their data structure, enabling interoperable reuse and facilitating collaboration.
**Open Science and FAIR Principles:** Throughout the project, students will adhere to open science principles, making all aspects of their work publicly available and reusable. This includes:

**Transparency**: All code, data, and workflow documentation will be openly accessible.
Reproducibility: The project will be designed to be reproducible, allowing others to build upon and extend the work.
**Interoperability**: The data structure interface will enable seamless reuse and integration with other projects.

By completing this project, students will gain hands-on experience with applying AI techniques to real-world challenges while embracing the principles of open science and FAIR research. The outcome will be a fully functional, FAIR, and open science-compliant project that showcases their skills and expertise.

## Course structure

The course is developed as a project. Students are divided into teams of 4-5 people and are assigned a raw dataset and a goal/task that requires to process, curate and build on the dataset.

The course is 30 hours, divided as per calendar below. The subdivision of these 30 hours is roughly as follows:

- First Lecture (2 hours): Presentation of the course, presentation of projects, team setup and dataset acquisition
- Dataset Exploration (2 hours)
- Strategy Discussion + First Exposition(2 hours)
- Strategy Implementation (8 hours)
- Second Exposition + Feedback (2 hours)
- Improvements + Secondary goals (8 hours)
- Repository + Project wrap up (4 hours)
- Final Exposition (2 hours)

## Datasets

Below we briefly describe the datasets involved

### Sensorium

Public dataset part of a Neuroscience competition consisting of the activity from the primary visual cortex of five different mice in response to around 700 videos each of length approx 10 seconds (approx 300 frames). The dataset includes additional behavioural measurements such as running speed, pupil dilation and eye movements.

Link: https://www.sensorium-competition.net/

![sensorium_2023_schematic-01 (1).png](attachment:bd31b0f3-12dd-44f0-9d93-473a34103713:sensorium_2023_schematic-01_(1).png)

**Dataset Structure**

Below we provide a brief explanation of the dataset structure and how to access all the information contained in them.

Have a look at our white paper for in depth description of the data. [White paper on arXiv](https://arxiv.org/abs/2305.19654)

We provide the datasets in the .zip format. Unzipping them will create two folders **data** and **meta**.

- **data:** includes the variables that were recorded during the experiment. The experimental variables are saved as a collection of numpy arrays. Each numpy array contains the value of that variable at a specific image presentation (i.e. trial). Note that the name of the files does not contain any information about the order or time at which the trials took place in experimental time. They are randomly ordered.
    - **videos:** This directory contains NumPy arrays where each single `X.npy` contains the video that was shown to the mouse in trial `X`.
    - **responses:** This directory contains NumPy arrays where each single `X.npy` contains the deconvolved calcium traces (i.e. responses) recorded from the mouse in trial `X` in response to the particular presented image.
    - **behavior:** Behavioral variables include pupil dilation and running speed. The directory contain NumPy arrays (of size `1 x 2`) where each single `X.npy` contains the behavioral variables (in the same order that was mentioned earlier) for trial `X`.
    - **pupil_center:** the eye position of the mouse, estimated as the center of the pupil. The directory contain NumPy arrays (of size `1 x 2`) for horizontal and vertical eye positions.
- **meta:** includes meta data of the experiment
    - **neurons:** This directory contains neuron-specific information. Below are a list of important variables in this directory
        - `cell_motor_coordinates.npy`: contains the position (x, y, z) of each neuron in the cortex, given in microns. **Note:** The
    - **statistics:** This directory contains statistics (i.e. mean, median, etc.) of the experimental variables (i.e. behavior, images, pupil_center, and responses).
        - **Note:** The statistics of the responses are or particular importance, because we provide the deconvolved calcium traces here in the responses.
        
        However, for the evaluation of submissions in the competition, we require the responses to be **standardized** (i.e. `r = r/(std_r)`).
        
    - **trials:** This directory contains trial-specific meta data.
        - `tiers.npy`: contains labels that are used to split the data into *train*, *validation*, and *test* set
            - The *training* and *validation* split is only present for convenience, and is used by our ready-to-use PyTorch DataLoaders.
            - The *test* set is used to evaluate the model preformance. In the competition datasets, the responses to all *test* images is withheld.

**Project Goals**

PROJECT 1

Primary: enrich the video files with all useful metadata (video type, repetitions, …)

Secondary:

- enrich the dataset with neural response and behavioural metadata building  visualization tools to navigate the dataset and to perform preliminary exploration

PROJECT 2
Primary: combine neural responses and videos with all the relevant metadata for an efficient query and organized data structure

Secondary:

- enrich the dataset with videos building visualization tools to navigate the dataset and to perform preliminary exploration

### SEM Images

This public dataset consists of large-scale Scanning Electron Microscopy (SEM) images covering 10 distinct categories of nanomaterials.

![SEM.png](attachment:165f04bd-d31e-44ee-b485-d2630647e691:SEM.png)

 The public version of this dataset contains processed images, while the raw experimental data resides in a separate, unorganized backup. The central challenge lies in reconstructing the data provenance using computer vision techniques, specifically by mapping the published JPEG images to their corresponding original TIFF files.

DOI: https://b2share.eudat.eu/records/e344a8afef08463a855ada08aadbf352

**Dataset Structure**

The data ecosystem for this project is divided into two distinct components: the organized public release and the raw experimental backup folder.

- **Public Data:** 25,537 JPEG images distributed across 10 categories, provided as `.tar` archives for each category (e.g., `Biological.tar`)
- **Raw Backup:** a collection of `.tiff` images acquired directly from the SEM microscope. All files are unorganized and lack inherent structure or labeling conventions.

**Project Goals**

The provenance link between the public `.jpg` and the original `.tiff` files has been lost; consequently, filenames do not correspond, and metadata associations are missing. 

Primary: start with the `Biological.tar` archive to extract the JPEG images and establish a valid mapping between the two data formats. A specific backup subset containing 10,000 unorganized **`.tiff`** images is provided for this development phase.

Secondary: extend the data provenance reconstruction to the entire published dataset, mapping all JPEG images from the public archives against the complete backup repository of over 100,000 `.tiff` files.
