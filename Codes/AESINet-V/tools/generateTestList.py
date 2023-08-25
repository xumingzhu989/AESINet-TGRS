#!/usr/bin/python
import os
import glob

rootpath = '/root/ILearnWork/ilData/'
datasets = ['ORSI']
for dataset in datasets:
    dataPath = rootpath + dataset + '/'
    outfile = rootpath + dataset + 'test_list.txt'

    with open(outfile,"w") as file:
        p_img = dataPath + 'Test-IMG/'
        imgFiles = glob.glob(os.path.abspath(os.path.join(p_img, '*.jpg')))
        imgFiles.sort()
        p_gt = dataPath + 'Test-GT/'
        gtFiles = glob.glob(os.path.abspath(os.path.join(p_gt, '*.png')))
        gtFiles.sort()

        Num_files = len(imgFiles)
        for i in range(Num_files):
            file.write(imgFiles[i].replace('\\','/') + ' ' 
            + gtFiles[i].replace('\\','/') + ' ' 
            + gtFiles[i].replace('\\','/') + 
            "\n")
