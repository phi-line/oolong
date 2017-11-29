## What is oolong?
This project aims to classify electronic music. It uses libraries like librosa, sci-kit image and sci-kit learn to collect data and form models that represent songs. Currently, a Kernel Density Estimator is used to represent a genre:

|  **House**  |  **Drum and Bass**  |  **Electro**  |
|:---:|:---:|:---:|
| ![mss-house_nearest2](https://user-images.githubusercontent.com/19956136/33244525-015fb872-d2ae-11e7-8290-764a0370581a.png) | ![mss-dnb_nearest2](https://user-images.githubusercontent.com/19956136/33244533-3a64ac90-d2ae-11e7-8487-4626ce2cce7e.png) | ![mos-electro_nearest2](https://user-images.githubusercontent.com/19956136/33244534-47983daa-d2ae-11e7-9746-52c742580fdd.png) |

## Analyzing a folder of songs
`TODO: add setup.py script`

First, clone the repository and install all dependencies. To start training you must have some songs of the same genre stored in a folder like: `audio/housePlaylist`

Oolong's main driver is `train.py`

    positional arguments:
        genre            classifier to train a model for
    optional arguments:
        -h, --help       show this help message and exit
        -l, --load       perform analysis on the input folder
        -t, --train      train a density model on the database`

Usage: `train.py genre folder`
> Running this example will create a database named with the folder name and a timestamp. The software will now start begin analyzing the songs in order and populate the database with their results. 

![train_output](https://raw.githubusercontent.com/phi-line/oolong/master/docs/images/train_output.png)

This process can take a while depending on the amount of songs you are scanning in. Each song takes ~20 seconds. The output database can be next used in training our density model.
