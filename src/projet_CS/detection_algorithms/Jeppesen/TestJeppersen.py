import csv
import sys
import re
import matplotlib.pyplot as plt
from Jeppesen import *


def test_jeppesen():
    pathcsv = 'C:/Users/paull/Projet-Epilepsie/seizure_detection_pipeline/src/usecase/00000258_s003_t002.csv'
    pathbi = 'C:/Users/paull/Projet-Epilepsie/seizure_detection_pipeline/src/usecase/00000258_s003_t002.tse_bi'
    k = 2
    assert type(jeppesen(pathcsv, pathbi, k)) == type([])


""""
path1 = 'rr_00007633_s003_t007.csv'
L = RRfetch(path1)
print(L)
"""
""""
path1 = '00000258_s003_t002.csv'
path2 = '00000258_s003_t002.tse_bi'
L1, L2 = noseizure(path1, path2)
print(L1, L2)
"""
"""
L = []
path2 = '00000258_s003_t002.tse_bi'
with open(path2) as csvfile:
    readtsebi = csv.reader(csvfile, delimiter=' ')
    for row in readtsebi:
        if row == 'version = tse_v1.0.0':
            pass
        else:
            L.append(float(','.join(row)))
print(L)
"""

if __name__ == "__main__":
    test_jeppesen()
