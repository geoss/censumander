# ACS_Regionalization

Code to reduce the margins of error in the American Community Survey (ACS) 
through intelligent regionalization.



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
clicking the "Download ZIP" button on this page or you can [fork
it](https://help.github.com/articles/fork-a-repo).


## Examples

We have built two examples to show the functionality of the code. One is a
very simple toy example on simulated data.  The second is a more complex
example based on the Austin metro area.  All the input data is included in the
repository.




