# Implementation of Scene Completion Using Millions of Photographs

This GitHub repository contains our implementation of James Hays's and Alexei Efros's [SIGGRAPH 2007](http://graphics.cs.cmu.edu/projects/scene-completion/) paper on scene completion. This served as our Fall 2022 CS 445 final project, for Sudharsan Krishnakumar Anitha, Luke Jacobs, and David Ho.

## Prerequisites

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install any necessary libraries used. 

Zip files containing the dataset we initially used can be found [here](https://drive.google.com/file/d/17e97oyOMjTLvtWbgoUnoV2KWPckOM3Yg/view?usp=sharing) (1.24 GB) and [here](https://drive.google.com/file/d/1CVjy_BWUkzs2hDsZLc79iAUu6D3UxeJy/view?usp=share_link) (5.18 GB). You will need to log in with an Illinois email to access these. Extract these zip files into a new `\dataset` subdirectory, called `\thousand-of-each` and `\5k-of-each`.

Test files, containing pairs of sample images with object masks can be found at the original paper's [website](http://graphics.cs.cmu.edu/projects/scene-completion/), under Data.

## Organization
`dataset` - Directory containing subdirectories of our database of images. `\thousand-of-each` contains one thousand images from five different categories, and `\5k-of-each` contains five thousand images from five different categories, as downloaded above. By default, as noted in `semantic_scene_matching.py`, we use the latter.

`results` - Directory containing sample results, as used in our final report.

`fill_image.py` - The main Python script for our project. Accepts three arguments: the original image we wish to patch, the mask representing the patch we wish to replace using the database, and the name of the output image.

`gist_descriptor.ipynb` - Test Jupyter notebook used to generate the gist descriptors for our database.

`keywords.txt` - List of keywords used to generate the databases.

`local_context_matching.py` - Python script which contains the functions used for the local context matching section, or finding the best patch out of the possible candidate images. 

`poisson_graph_cut.py` - Python script which contains the functions used for graph cut segmentation and final Poisson blending.

`script.py` - Python script used to generate the databases.

`semantic_scene_matching.ipynb` - `semantic_scene_matching.py`, as a Jupyter notebook.

`semantic_scene_matching.py` - Python script which contains the functions used for the semantic scene matching section, or choosing the best candidate images from our database. 

## How to Use

Assuming an input image and corresponding object mask has been placed into the main directory (named test.bmp and test_mask.bmp, respectively), the main script can be run using:

```bash
python fill_image.py "test.bmp" "test_mask.bmp" "test_output.png"
```
