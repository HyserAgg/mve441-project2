#! /bin/bash

DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"
COMPRESSED_SOURCE="TCGA-PANCAN-HiSeq-801x20531.tar.gz"

echo "Downloading data.."
wget $DATA_URL

echo "Extracting data"
tar -xf $COMPRESSED_SOURCE

echo "Done!"