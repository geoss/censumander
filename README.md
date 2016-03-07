# Reducing the Margins of Error in Census Tract and Block Group Data from the American Community Survey

The American Community Survey (ACS) is the largest survey of US households and is the principal source for neighborhood scale information about the US population and economy. The ACS is used to allocate billions in federal spending and is a critical input to social scientific research in the US. However, estimates from the ACS can be highly unreliable. For example, in over 72% of census tracts, the estimated number of children under 5 in poverty has a margin of error greater than the estimate. Uncertainty of this magnitude complicates the use of social data in policy making, research, and governance. This article presents a heuristic spatial optimization algorithm that is capable of reducing the margins of error in survey data via the creation of new composite geographies, a process called regionalization. Regionalization is a complex combinatorial problem. Here rather than focusing on the technical aspects of regionalization we demonstrate how to use a purpose built open source regionalization algorithm to process survey data in order to reduce the margins of error to a user-specified threshold.

This repository includes code that reduces the margins of error in ACS Tract and Block Group Level Data by "intelligently" combining Census geographies together into regions.  A region is a collection of 1 or more census geographies that meets a user specified margin of error (or CV).  We refer to this procedeure as "regionalization."  

Technical details of this paper and example implementations are described in this [PLOSOne Paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0115626#abstract0).


## Getting Started

### Prerequisites

All the scripts are written for [Python](http://www.python.org/) 2.7 (earlier
versions have not been tested). We recommend installing [Anaconda
python](https://www.continuum.io/downloads) as this distribution provides easy access to all the necessary libraries to run the code. There are a dependencies on the following libraries.  

* [Numpy](http://www.scipy.org/install.html) 1.3 or later
* [Scipy](http://www.scipy.org/install.html) 0.7 or later
* [PySAL](http://pysal.org) 1.5 or later
* [pandas] (http://pandas.pydata.org) 0.11.0 or later
* [MDP](http://mdp-toolkit.sourceforge.net) 3.2 or later
* [Bottleneck](https://pypi.python.org/pypi/Bottleneck) 0.7 or later


##Examples
We have built two [Jupyter Notebooks](https://jupyter.org/) to show the
functionality of the code.  The notebooks and all input data needed to run them are
included in the repository.  The notebooks require the
[matplotlib](http://matplotlib.org/), [shapely](http://toblerity.org/shapely/) and [geopandas](http://geopandas.org/) packages for the visulaizations. Static versions can be viewed from the following links.

* [Toy Example](http://nbviewer.ipython.org/github/geoss/ACS_Regionalization/blob/master/code/toy_example.ipynb)
  is a very simple example on simulated data.

* [Austin Example](http://nbviewer.ipython.org/github/geoss/ACS_Regionalization/blob/master/code/austin.ipynb)
  is a more complex example using data from the Austin metro area.






