#!/usr/bin/env bash
awk -F, '$3 !~ /Training/' fer2013.csv | cut -d, -f1,2 > valid.csv;
awk -F, '$3 ~ /Training/' fer2013.csv | cut -d, -f1,2 >> train.csv