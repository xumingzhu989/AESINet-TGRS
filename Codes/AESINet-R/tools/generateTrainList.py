import os
import glob

rootpath = '/root/ZengXiangyu/AESINetData/'
datasets = ['ORSI16000','EORSSD11200','ORSSD4800']

for dataset in datasets:
    dataPath = rootpath + dataset + '/'
    outfile = rootpath + dataset + 'train_list.txt'

    with open(outfile,"w") as file:
        p_img = dataPath + 'Image/'
        imgFiles = glob.glob(os.path.abspath(os.path.join(p_img, '*.jpg')))
        imgFiles.sort()
        p_gt = dataPath + 'GT/'
        gtFiles = glob.glob(os.path.abspath(os.path.join(p_gt, '*.png')))
        gtFiles.sort()
#        p_edge = dataPath + 'EdgeGT/'
#        edgeFiles = glob.glob(os.path.abspath(os.path.join(p_edge, '*.png')))
#        edgeFiles.sort()

        Num_files = len(imgFiles)
        for i in range(Num_files):
            file.write(imgFiles[i].replace('\\','/') + ' ' 
            + gtFiles[i].replace('\\','/') + ' ' 
            + gtFiles[i].replace('\\','/') + 
            "\n")

