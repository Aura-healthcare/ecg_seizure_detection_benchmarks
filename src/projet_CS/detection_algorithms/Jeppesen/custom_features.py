from cmath import nan
import json
import csv
from statistics import median
import sys
import re
import time
import matplotlib.pyplot as plt
import random as rd
import numpy as np

from hrvanalysis import (
    get_time_domain_features,
    get_geometrical_features,
    get_frequency_domain_features,
    get_csi_cvi_features,
    get_poincare_plot_features,
)

def filter_median(rr_intervals_list,k=7):
    """
    Applies a k-long median filter to the RR intervals list in entry
    To each index is assigned the median of k previous RR interval lengths
    The objective is to counter errors in the ECG reading, 
    such as missing a heartbeat or couting two instead of one

    Parameters
    ----------
    rr_intervals_list : the liste of rr_interval lengths to be filtered (unit doesn't matter)
    k (optional) : medians will be calculated on lists of k consecutive RR interval lengths


    Returns
    -------
    rr_intervals_list_filtered : the filtered list of RR intervals lengths

    """
    rr_intervals_list_filtered=rr_intervals_list[0:k]
    for i in range(k,len(rr_intervals_list)):
        rr_intervals_list_filtered.append(median(rr_intervals_list[i-k:i]))
    return rr_intervals_list_filtered

def reg_Lin(x,y):
    #Note : ce code a été copié-collé depuis https://gsalvatovallverdu.gitlab.io/python/moindres-carres/
    """
    Ajuste une droite d'équation a*x + b sur les points (x, y) par la méthode
    des moindres carrés.

    Args :
        * x (list): valeurs de x
        * y (list): valeurs de y

    Return:
        * a (float): pente de la droite
        * b (float): ordonnée à l'origine
    """
    # conversion en array numpy
    x = np.array(x)
    y = np.array(y)
    # nombre de points
    npoints = len(x)
    # calculs des parametres a et b
    a = (npoints * (x*y).sum() - x.sum()*y.sum()) / (npoints*(x**2).sum() - (x.sum())**2)
    b = ((x**2).sum()*y.sum() - x.sum() * (x*y).sum()) / (npoints * (x**2).sum() - (x.sum())**2)
    # renvoie des parametres
    return a, b


def slope(rr_intervals_list):
    """
    Gets the slope of the tachygram (lengths of rr intervals in fonction of their index)

    Parameters
    ----------
    rr_intervals_list : the liste of rr_interval lengths

    Returns
    -------
    a : the slope of the tachygram (can be negative)

    """
    abcisse=[]
    for i in range(1,len(rr_intervals_list)+1):
        abcisse.append(i)
    a,b=reg_Lin(abcisse,rr_intervals_list)
    return a

def get_custom_features(rr_intervals_list):
    """
    This function calculates custom features from a RR intervals list
    It is convenient if you want to add your own features
    Already coded are the features evocated in the scientific article Aura transmitted us

    Parameters
    ----------
    rr_intervals_list : the liste of rr_interval lengths whose custom features will be extracted from


    Returns
    -------
    returned_dic : dictionnary ---> keys are features names
                               ---> returned_dic['feature_name'] is the value of the specified feature
                                    on the spectified RR intervals list

    """

    returned_dic={}

    #Calcul des features modCSI * slope et CSI*slope
    csi_cvi_features=get_csi_cvi_features(rr_intervals_list)
    csi = csi_cvi_features['csi']
    modified_csi=csi_cvi_features['Modified_csi']
    actual_slope=slope(rr_intervals_list)
    

    returned_dic["Modified_csi*Slope"]=modified_csi*abs(actual_slope)
    returned_dic["csi*Slope"]=csi*abs(actual_slope)

    return returned_dic


    

