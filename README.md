# m2mcluster
A python packaged for made to measure modelling of star clusters. Please cite Webb, Hunt, and Bovy 2022 if you use this package for your work.

# Installation
To install m2mcluster from GitHub, clone the repository and install via setup tools:

`git clone https://github.com/webbjj/m2mcluster.git`
`cd m2mcluster`
`python setup.py install`

Please note that if you don’t have permission to write files to the default install location (often /usr/local/lib), you will either need to run:

`sudo python setup.py install`

or

`python setup.py install --prefix='PATH'`

where ‘PATH’ is a directory that you do have permission to write in and is in your PYTHONPATH.

# Requirements

m2mcluster requires the following python packages:[amuse-framework](https://amuse.readthedocs.io/en/latest/index.html), [galpy](https://docs.galpy.org/en/v1.8.1/), [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/)

# Examples

The examples directory contains a generic script that highlights how m2mcluster is used and shows all the relevant input parameters. There are also two examples taken from Webb, Hunt, and Bovy 2022.
