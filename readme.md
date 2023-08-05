## What is oolong?
This project aims to generate training data for music genre classification. By generating music ML training data though the use of computer vision feature fingerprinting techniques, we can do things like:
* retrieve based on matched fingerprints (similar to Shazam, Soundhound, etc.)
* coallece those fingerprints to understand patterns of features, unique to a genre of music

I generated features for the detected "chorus" of electronic songs found on the internet. Here's what those features look like, coallesced into a Kernel Density Estimation, colored by density:

|  **House**  |  **Electro**  |  **Drum and Bass**  |
|:---:|:---:|:---:|
| ![mss-house](https://raw.githubusercontent.com/phi-line/oolong/master/docs/images/mss-house_kde.png) | ![mos-electro](https://raw.githubusercontent.com/phi-line/oolong/master/docs/images/mos-electro_kde.png) | ![mss-dnb](https://raw.githubusercontent.com/phi-line/oolong/master/docs/images/mss-dnb_kde.png)


## Setup and Usage
`TODO: add setup.py script`

First, clone the repository and install all dependencies. To start training you must have some songs of the same genre stored in a folder like: `audio/housePlaylist`

Oolong's main driver is `train.py`

    positional arguments:
        genre            classifier to train a model for
    optional arguments:
        -h, --help       show this help message and exit
        -l, --load       perform analysis on the input folder
        -t, --train      train a density model on the database

### Examples:

**Analyzying songs**:

Usage: `train.py genre --load folder`
> Running this example will create a database named with the folder name and a timestamp. Oolong will now start begin analyzing the songs in order and populate the database with their results. <!--More info-->

![train_output](https://raw.githubusercontent.com/phi-line/oolong/master/docs/images/train_output.png)

This process can take a while depending on the amount of songs you are scanning in. Each song takes ~20 seconds. Each song will fill the nested class structure found in `song_classes.py`. The output database will be a collection of songs stored in JSON format.

**Generating a density model for a genre**:

Usage: `train.py genre --train db_path`
> Running this example will read in from the database JSON. Oolong will now generate a Kernel Density Estimator and display it back to the user. <!--More info-->

Data that is larger than the average size of a song (in frames) will be thrown away. For the density calculation, a fixed x and y size is required. This is one of the core issues of this approach and is talked about in detail below.

<!--## Analyzing a group of songs-->

## Training the density model
> Wait? A Kernel Density What?

A [Kernel Density Estimator](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation) can be seen as a heatmap of scatterplot data. It is most comparable to a histogram in that it has bins to seperate data - however, it's goal is to not only model 

<table>
    <tr><td><img alt="open opps 1" src="https://raw.githubusercontent.com/phi-line/oolong/master/docs/images/smb-slice_kde.png"></td></tr>
    <tr><td>This is a density model created from the features of a single song. Visually, you can see 'bins' of data forming for each kick snare pattern.</td></tr>
</table>

The idea is that if we combine the features of every single song's most significant segement, the output density model will be representative of the entire genre. <!--As is stands there are some core problems with this approach: -->

