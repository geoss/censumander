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


We recommend installing [Anaconda
python](https://store.continuum.io/cshop/anaconda/) as it bundles python and
all the necessary libraries in one download, except
[Bottleneck](https://pypi.python.org/pypi/Bottleneck).  An alternative is
the [Enthought Python Distribution
(EPD)](https://www.enthought.com/products/epd/); the academic and pay versions
have all the libraries except [PySAL](http://pysal.org), while the free
version requires [PySAL](http://pysal.org),
[MDP](http://mdp-toolkit.sourceforge.net) and
[Bottleneck](https://pypi.python.org/pypi/Bottleneck) to be installed
separately. 


### ACS Regionalization Code

You can download the actual regionalization code (i.e. this repository) by
clicking the [Download
ZIP](https://github.com/geoss/ACS_Regionalization/archive/master.zip) button
on this page or you can [fork
it](https://help.github.com/articles/fork-a-repo).


## Examples

We have built two [IPython Notebooks](http://ipython.org/notebook) to show the
functionality of the code.  All the input data needed to run the notebooks is
included in the repository.  The notebooks require the
[matplotlib](http://matplotlib.org/) library and [R](http://www.r-project.org)
for the visulaizations.

* [Toy Example](http://nbviewer.ipython.org/github/dfolch/map_test/blob/master/toy_example.ipynb?create=1)
  is a very simple example on simulated data.

* [Austin Example](http://nbviewer.ipython.org/github/dfolch/map_test/blob/master/austin.ipynb?create=1)
  is a more complex example using data from the Austin metro area.





