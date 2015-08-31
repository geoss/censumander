# ACS Regionalization

This repository includes code that reduces the margins of error in ACS Tract and Block Group Level Data by "intelligently" combining Census geographies together into regions.  A region is a collection of 1 or more census geographies that meets a user specified margin of error (or CV).  We refer to this procedeure as "regionalization."  



## Installation

### Prerequisites

All the scripts are written for [Python](http://www.python.org/) 2.7 (earlier
versions have not been tested). There are a number of additional dependencies
that need to be installed.  

* [Numpy](http://www.scipy.org/install.html) 1.3 or later
* [Scipy](http://www.scipy.org/install.html) 0.7 or later
* [PySAL](http://pysal.org) 1.5 or later
* [pandas] (http://pandas.pydata.org) 0.11.0 or later
* [MDP](http://mdp-toolkit.sourceforge.net) 3.2 or later
* [Bottleneck](https://pypi.python.org/pypi/Bottleneck) 0.7 or later


We recommend installing [Enthought Canopy](https://www.enthought.com/products/canopy/) or [Anaconda
python](https://store.continuum.io/cshop/anaconda/) as these two distributions bundle python and
provide access to all the necessary libraries to run the code.


### ACS Regionalization Code

You can download the actual regionalization code (i.e. this repository) by
clicking the [Download
ZIP](https://github.com/geoss/ACS_Regionalization/archive/master.zip) button
on this page or you can [fork
it](https://help.github.com/articles/fork-a-repo).


## Examples

We have built two [Jupyter Notebooks](https://jupyter.org/) to show the
functionality of the code.  The notebooks and all input data needed to run them are
included in the repository.  The notebooks require the
[matplotlib](http://matplotlib.org/), [shapely](http://toblerity.org/shapely/) and [geopandas](http://geopandas.org/) packages for the visulaizations. Static versions can be viewed from the following links.

* [Toy Example](http://nbviewer.ipython.org/github/geoss/ACS_Regionalization/blob/master/code/toy_example.ipynb)
  is a very simple example on simulated data.

* [Austin Example](http://nbviewer.ipython.org/github/geoss/ACS_Regionalization/blob/master/code/austin.ipynb)
  is a more complex example using data from the Austin metro area.





