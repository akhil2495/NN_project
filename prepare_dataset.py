import os
import pandas as pd
from shutil import copyfile

directory = 'TUT-urban-acoustic-scenes-2018-development'

with open('data/'+directory+'.meta/'+directory+'/meta.csv') as f:
    df = pd.read_csv(f, delimiter='\t')
    
    cities = []
    for city in df['identifier']:
        cities.append(city.split('-')[0])
    cities = list(set(cities))

    train_cities = cities[2:]
    val_cities = cities[1]
    test_cities = cities[0]

    train_files = []
    test_files = []
    val_files = []
    for index, row in df.iterrows():
        if row['identifier'].split('-')[0] in train_cities:
            train_files.append(row['filename'])
        if row['identifier'].split('-')[0] in val_cities:
            val_files.append(row['filename'])
        if row['identifier'].split('-')[0] in test_cities:
            test_files.append(row['filename'])


    print('Started')
    for src in train_files:
        dst = src.split('/')[1]
        copyfile(src, 'dataset/train/' + dst)
    print('1')
    for src in val_files:
        dst = src.split('/')[1]
        copyfile(src, 'dataset/val/' + dst)
    print('2')
    for src in test_files:
        dst = src.split('/')[1]
        copyfile(src, 'dataset/test/' + dst)
