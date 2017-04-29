#!/bin/bash

url=https://raw.githubusercontent.com/CNevd/datasets/master
file=binarySentiment.zip
if [ ! -e data/binarySentiment ]; then
        wget ${url}/${file} -P data/
        cd data
        unzip ${file}
    else
        echo "data/binarySentiment already exits"
fi
