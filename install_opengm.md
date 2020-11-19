### Compile and intall OpenGM with python wrapper in virtualenv 

#### Online references
- conda install guide:
	- https://groups.google.com/forum/#!searchin/opengm/nose%7Csort:relevance/opengm/Nte5Zpu9RL0/YSanK09kNwAJ
- plain ubuntu install guides:
	- http://cvlab-dresden.de/HTML/people/bogdan/teaching/slides-script/ml2-ss15/installation-readme.txt
	- https://memoryaux.wordpress.com/2014/08/15/installing-opengm-with-python-wrapper/

#### Instructions (tested for Ubuntu 14.04)

clone source using  
`git clone https://github.com/opengm/opengm.git`

make build dir under opengm/ 

`makedir build/`

and enter build/

`cd build/`

using ccmake and try to configure with 'c' 

`ccmake ../`

run ccmake again and select options

`ccmake ../`


build:
  - command line ?
  - converter ?
  - docs ?
  - examples ? (requires external lib like cplex)
  - python docs ? (requires pip install sphinx and produces ugly outputs)
  - python wrapper
  - testing
  - tutorials

with:

  - boost
  - hdf5

python:

  - python exectuable: /home/USER/.virtualenvs/VNAME/bin
  - include dir: /home/USER/.virtualenvs/VNAME/include
  - include dir2: /home/USER/.virtualenvs/VNAME/include/python2.7
  - library: /usr/lib/x86_64-linux-gnu/libpython2.7.so 
(alternative is /home/USER/.virtualenvs/VNAME/lib/python2.7, but no *.so file here)
  - library debug: PYTHON_LIBRARY_DEBUG-NOTFOUND (default)
  - numpy include directory: /home/USER/.virtualenvs/VNAME/lib/python2.7/site-packages/numpy/core/include

*for some unkown reason* opengm python site-package is installed under `/usr/local/lib/python0./`
therefore, better to skip make install and simply copy files by hand (see below)


To build run (-j only if multicore system):

```
make -j4
make -j2 test
make install

```


simply copy it to `/home/USER/.virtualenvs/VNAME/lib/python2.7/site-packages/`

now test in python:
`import opengm`

Hopefully things work :)