## Setting Up Torch Geo For Distributed

This process can be slightly confusing on the lab machines, but I'll try to give the basics.

### Python Virtual Environments
First you should install python virtual environments so that the torch install goes smoothly.
To do this you should make a python_virtual environments directory somewhere, mines in my home dir.
Change into this directory.

First install python virtual environments module to your user on the cs machines.
```
pip install --user virtualenv
```

Then create a virtual environment for the torch install in your virtual environments directory.

```
python -m virtualenv torch
```

Source this environment with the following command

```
cd /path/to/pyenvironments
source torch/bin/activate
```

Install the dependencies in requirements.txt
```
pip install -r requirements.txt
```

Then you need to export the environment for the cuda devices on the cs machines. We can't use these because they are too old,
but pytorch detects them and thinks it's using cuda anyways.

```
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Finally you should have everything you need to get the distributed stuff to run. The file takes two args:
The first is the rank of the program (the master is rank 0, the rest are in ascending order it shouldn't matter too much).
The second arg is the world_size which is the number of workers. So if you are running it on 4 machines, you should have ranks [0,1,2,3] and a world_size of 4.

The program will wait to every node is online to start training. Currently the model doesn't do much of anything but you can print out the losses in the training program if you want to see what's happening.
