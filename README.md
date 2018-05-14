# MSMadapter
[![Build Status](https://travis-ci.org/jeiros/msmadapter.svg?branch=master)](https://travis-ci.org/jeiros/msmadapter)
[![Coverage Status](https://coveralls.io/repos/github/jeiros/msmadapter/badge.svg?branch=master)](https://coveralls.io/github/jeiros/msmadapter?branch=master)

Implement an adaptive search for MD simulations run with AMBER.

Use your own msmbuilder defined models for the search.

# Installation
Clone and install from source
```bash
git clone https://github.com/jeiros/msmadapter.git
cd msmadapter
pip install -e .
```

# Example
A crude example of the necessary directory structure to get started:

```
.
├── [  50]  generators
│   ├── [  66]  gen1
│   │   ├── [ 575]  Production.in
│   │   ├── [  33]  seed.ncrst
│   │   └── [  36]  structure.prmtop
│   ├── [  66]  gen2
│   │   ├── [ 575]  Production.in
│   │   ├── [  33]  seed.ncrst
│   │   └── [  36]  structure.prmtop
│   ├── [  66]  gen3
│   │   ├── [ 575]  Production.in
│   │   ├── [  33]  seed.ncrst
│   │   └── [  36]  structure.prmtop
│   └── [  66]  gen4
│       ├── [ 575]  Production.in
│       ├── [  33]  seed.ncrst
│       └── [  36]  structure.prmtop
└── [ 523]  msmadapt.py
```

Where `msmadapt.py` is the script that controls the adaptive search logic.

A lot of it can be left as defaults:
```python
from msmadapter.adaptive import App, Adaptive
app = App(from_solvated=True)
ad = Adaptive(app=app, stride=1, atoms_to_load='not water',
    sleeptime=6*3600)
ad.run()
```
Then start the adaptive sampling scheme with the following command:
```bash
nohup python msmadapt.py
```
The `nohup` is to be able to detach from the computer where you're running this.
You can trace the output of the search on the contents of the `nohup.out` file that will be generated.