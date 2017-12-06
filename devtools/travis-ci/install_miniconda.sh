#!/bin/bash
MINICONDA=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_MD5=$(curl -s https://repo.continuum.io/miniconda/ | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')
wget https://repo.continuum.io/miniconda/$MINICONDA
if [[ $MINICONDA_MD5 != $(md5sum $MINICONDA | cut -d ' ' -f 1) ]]; then
    echo "Miniconda MD5 mismatch"
    exit 1
fi
bash $MINICONDA -b
rm -f $MINICONDA

export PATH=$HOME/miniconda3/bin:$PATH

conda install -yq python=$CONDA_PY
conda update -yq conda
conda config --add channels omnia
conda install scipy cython pandas matplotlib -yq
conda install -yq conda-build jinja2
conda install msmbuilder msmexplorer -yq
conda install ambertools=17 -c http://ambermd.org/downloads/ambertools/conda/ -yq
