# Subdataset 1 Codes

This directory contains code for meshing, FEA, example domain generation and ML

For the domain files (subdataset1_geometry.zip):

x.txt indicates the centers of each block and l.txt gives the length of each block. Each block is stacked top to bottom. Note that the coordinates for all domains are stored as image arrays where the origin is in the top left. The coordinates have to flipped accordingly to generate the domain for meshing.

## Meshing (subdataset1_mesh.py)
This code generates mesh to use in FEA. 
If using the directory to run code, we recommend to download the geometry dataset (subdataset1_geometry) and place the file here in this directory.

The code is run with the following versions:

* Python version: 3.7.7
* Gmsh version: 4.6.0

## Image generation (subdataset1_img.py)

This is the code to generate the domain geometry as an image from the files in subdataset1_geometry
The code is run with the following versions:

* Python version: Any should work
* Skimage: 0.18.1
