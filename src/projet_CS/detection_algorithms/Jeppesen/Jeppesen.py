from cmath import nan
import csv
import json
import numpy as np
import argparse
import sys

from hrvanalysis import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_csi_cvi_features,
    get_poincare_plot_features,
)

from detection_algorithms.Jeppesen.custom_features import get_custom_features


def RRfetch(file):
    """
    From a CSV.RR file extract the RR interval list

    Parameters
    ----------
    csv.RR : Mesure of the RR intervall

    Returns
    -------
    RRlist : list of RR intervals
    ex : [820,790,810]
    """
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        RRlist = []
        for row in readCSV:
            if row[2] == "rr_interval":
                pass
            else:
                RRlist.append(float(row[2]))
    return RRlist


def noseizure(tsebi, csvRR):
    """
    Select sample of ECG without any seizures in order to calculate floor values

    Parameters
    ----------
    tsebi : Frame with begining/end seizure, seiz or no seiz and probability
    csvRR : Mesure of the RR intervall

    Returns
    -------
    noseizure : list of time period of ECG without seizure in second
    ex : [13,37]
    """

    noseiztime = []
    with open(tsebi) as csvfile:
        readtsebi = csv.reader(csvfile, delimiter=' ')
        for row in readtsebi:
            if row == ['version', '=', 'tse_v1.0.0']:
                pass
            elif row == []:
                pass
            elif row[2] == 'bckg':
                noseiztime = [float(row[0])*(10**3),
                              float(row[1])*(10**3)-60*(10**3)]
                return(noseiztime)


def extract_RR_noseizure(jsonpath, csvRR):
    """
    Using sample of ECG without any seizures we extract the associated RR
    values in that frame

    Parameters
    ----------
    noseiztime : sample of ECG without any seizures
    csvRR : Mesure of the RR intervall

    Returns
    -------
    noseiz : list of RR intervals without seizure
    ex : [820,790,810]
    """

    RR = RRfetch(csvRR)
    noseiztime=noseizurejson(jsonpath,csvRR)
    debutRR = RR[0]
    finRR = RR[0]
    time_begin_index = 0
    time_finish_index = 0
    i = 0
    j = 0

    while debutRR < noseiztime[0]:
        i += 1
        debutRR += RR[i]
    time_begin_index = i

    while finRR < noseiztime[1]:
        j += 1
        finRR += RR[j]
    time_finish_index = j

    noseiz = RR[time_begin_index:time_finish_index]
    return(noseiz)


def noseizurejson(json_path, csvRR):
    """
    Select sample of ECG without any seizures to calculate floor values

    Parameters
    ----------
    json : Json file associated with csv
    csvRR : Csv file of the mesure of the RR intervall

    Returns
    -------
    noseizure : list of time period of ECG without seizure in second
    ex : [13,37]

    """
    noseiztime = []
    with open(json_path) as json_data:
        data_dict = json.load(json_data)
        noseizure = data_dict['background'][0]
    noseizure = [noseizure[0]*(10**3), noseizure[1]*(10**3)-60*(10**3)]
    return(noseizure)


"""
==================================================================================================
The following code isn't part of a function. It is necessary for the functionning
of the next 2 functions.
Its purpose is to list every feature, and have a link between a feature's name and its index.
"""

# This list is completely random, there's no meaning behind these values.
# Its only purpose is to be able to use the following functions
rr_intervals_list_test = [
    739, 711, 731, 701, 895, 729, 709, 716, 767, 779, 890, 748, 821, 702, 831, 829, 840, 817, 703, 866, 804, 744, 899, 792, 720, 737, 805, 792, 775, 704, 735,
    742, 821, 882, 896, 803, 864, 790, 842, 772, 828, 799, 808, 717, 833, 701, 826, 721, 784, 886, 727, 767, 894, 846, 894, 855, 832, 853, 798, 770, 754, 892, 811, 705, 761, 830, 874, 878, 883, 852, 869, 854, 750, 899,
    895, 824, 717, 879, 811, 833, 864, 731, 761, 899, 736, 898, 835, 771, 740, 783,
    856, 828, 768, 727, 736, 837, 721, 739, 779, 703, 866, 772, 875, 756, 752, 852, 870, 798, 880, 829, 883, 763, 895, 823, 715, 900,
]

#Creating dictionnaries containing features and their values
timestamps = {}
timestamps['timestamp']=0 #La première ligne du dictionnaire n'est pas une feature, 
#elle contiendra les timestamps de chaque intervalle RR, pour des raisons pratiques
time_domain_features = get_time_domain_features(rr_intervals_list_test)
frequency_domain_features = get_frequency_domain_features(rr_intervals_list_test)
csi_cvi_features = get_csi_cvi_features(rr_intervals_list_test)
poincare_plot_features = get_poincare_plot_features(rr_intervals_list_test)
custom_features = get_custom_features(rr_intervals_list_test)


#Gros dictionnaire (fusion des 6 précédents)
noms_features = dict(
    (key, value)
    for d in (
        timestamps,
        time_domain_features,
        # geometrical_features,
        frequency_domain_features,
        csi_cvi_features,
        poincare_plot_features,
        custom_features
    )
    for key, value in d.items()
)

#The dictionnary 'nom_features' is now becoming the link between a feature's name and its index
#Example : noms_features['Modified_csi*Slope']==30
i=0
for cle in noms_features:
    noms_features[cle]=i
    i+=1

#You can print the dictionnary to have access to all available features

#print(noms_features)

"""
This part is now over. We can now head to the functions that will calculate the values of features.
==================================================================================================
"""


def calcul_features(rr_intervals_list, sliding_window=100):
    """
    Gets the value of all features on the specified RR intervals list.
    There is a sliding window, whose value can for example be a hundred.
    Then, for each RR interval whose index is superior to 100, the values of features will be calculated
    on the list of the 100 previous RR intervals.
    This allows us to have the values of features depend on time
    These values will be stocked into a matrix


    Parameters
    ----------
    rr_intervals_list : a list containing all RR interval lengths
    sliding_window (optional) : number of RR intervals that will be the basis of the feature values

    Returns
    -------
    matrice_features :
        One line for each feature (first line is the timestamps of each RR interval)
        One column for each RR interval

        As we can't reach the size of the sliding_window for the first 99 RR intervals (in our example),
        we put the value 'nan' instead in the matrix, which means that the value doesn't exist.
    """

    L = len(noms_features)
    # Total number of features (including the timestamps)

    matrice_features = np.zeros((L, len(rr_intervals_list)))

    # Let's focus on the first line (entering the timestamps)
    s = 0
    for j in range(len(rr_intervals_list)):
        matrice_features[0, j] = s
        s += rr_intervals_list[j]

    # Imaginons que sliding_window = 100. Alors pendant les 100 premiers intervalles rr,
    # on ne pourra pas calculer les features. Donc on met des nan à la place
    for j in range(0, sliding_window):
        for i in range(1, L):
            matrice_features[i][j] = nan


    """
    Right now is the important part : calculation of the features values
    """
    for t in range(sliding_window, len(rr_intervals_list)):

        time_domain_features = get_time_domain_features(
            rr_intervals_list[t - sliding_window: t])
        frequency_domain_features = get_frequency_domain_features(
            rr_intervals_list[t - sliding_window: t])
        csi_cvi_features = get_csi_cvi_features(
            rr_intervals_list[t - sliding_window: t])
        poincare_plot_features = get_poincare_plot_features(
            rr_intervals_list[t - sliding_window: t])
        custom_features = get_custom_features(
            rr_intervals_list[t - sliding_window: t])

        #Fusion des dictionnaires
        features = (
            time_domain_features
            | frequency_domain_features
            | csi_cvi_features
            | poincare_plot_features
            | custom_features
        )

        # Le calcul des features à t se base sur les 100 (sliding_window) intervalles RR précédents

        for cle in features.keys():
            indice_feature = noms_features[cle]
            # print(indice_feature, cle)
            matrice_features[indice_feature, t] = features[cle]
    return matrice_features


def calcul_seuils(rr_intervals_list_bckg,sliding_window=100):
    """
    Gets the values of all features thresholds on the specified RR intervals list.
    This threshold shall be calculated on a period without seizures : that's why we decided 
    to take the recording into account only until the minute before the first seizure (if there is)

    Parameters
    ----------
    rr_intervals_list_bckg : the preselected list containing all RR interval lengths, on a period without seizures
    sliding_window (optional) : number of RR intervals that will be the basis of the feature values

    Returns
    -------
    liste_seuils :
        Its size is as long as the number of features
        For each feature, liste_seuils[index_feature] contains the value of the associated threshold
    """

    liste_seuils = [nan]
    # On cherche la valeur max de chaque feature
    matrice = calcul_features(rr_intervals_list_bckg,sliding_window)
    for i in range(1, len(noms_features)):
        liste_valeurs = matrice[i][:]
        liste_valeurs = [
            valeur for valeur in liste_valeurs if not (np.isnan(valeur))]
        liste_seuils.append(max(liste_valeurs))
    return liste_seuils


def condition_exceeding_the_threshold_1(features,k,timestamp,threshold,timestamps_beginning_seizures): # threshold crossing condition and dead zone of 3 minutes after a seizure where a new one cannot be detected
    if abs(features[k][timestamp]) >= 1.05*abs(threshold[k]) and features[0][timestamp]-timestamps_beginning_seizures[len(timestamps_beginning_seizures)-1] >= 180000:
        return True
    else :
        return False
    


def condition_exceeding_the_threshold_2(features,k,timestamp,threshold,timestamps_beginning_seizures,is_timestamp_during_a_seizure,index): # another condition for the moment to be detected as being during a seizure
    if abs(features[k][timestamp]) >= 1.05*abs(threshold[k]) and (features[0][timestamp]-timestamps_beginning_seizures[len(timestamps_beginning_seizures)-index] >= 180000 or is_timestamp_during_a_seizure[timestamp-1][0] == 1):
        return True
    else:
        return False
    



def jeppesen(enregistrement_path, json_path, feature_name ,sliding_window=100):
    k=noms_features[feature_name] #The index of the selected feature
    # define the variables from the previously established code portions
    n = extract_RR_noseizure(json_path, enregistrement_path)
    r = RRfetch(enregistrement_path)
    threshold = calcul_seuils(n,sliding_window)
    features = calcul_features(r,sliding_window)
    
    
    timestamps_beginning_seizures = [-180000]  # this first coefficient always allows to detect the first seizure and this matrix will include the moments of beginning of seizure
    is_timestamp_during_a_seizure = [[0, 0]]  # the second column of this matrix will correspond to the times and the first column to whether this moment is during a seizure or not (0 or 1)
    StartTime_and_EndTime_seizures = [0]  # this row matrix corresponds to the start and end dates of each seizure (transition between background and seizure periods)
    is_period_a_seizure = []  # this list will contain lists indicating the start time of a period, the end time and if this period is a crisis or not (0 or 1)
    
    
    timestamps=range(1, np.shape(features)[1])
    
    
    for timestamp in timestamps:
        if features[k][timestamp] != nan:
            if condition_exceeding_the_threshold_1(features,k,timestamp,threshold,timestamps_beginning_seizures):
                timestamps_beginning_seizures.append(features[0][timestamp])
                if condition_exceeding_the_threshold_2(features,k,timestamp,threshold,timestamps_beginning_seizures,is_timestamp_during_a_seizure,2):
                    is_timestamp_during_a_seizure.append([1, features[0][timestamp]])
            elif condition_exceeding_the_threshold_2(features,k,timestamp,threshold,timestamps_beginning_seizures,is_timestamp_during_a_seizure,1):
                is_timestamp_during_a_seizure.append([1, features[0][timestamp]])
            else:  # otherwise the moment is detected as not having a seizure
                is_timestamp_during_a_seizure.append([0, features[0][timestamp]])
    
    
    
    for timestamp in timestamps:
        # this loop allows to build StartTime_and_EndTime_seizures as previously defined
        if (is_timestamp_during_a_seizure[timestamp-1][0] == 1 and is_timestamp_during_a_seizure[timestamp][0] == 0) or (is_timestamp_during_a_seizure[timestamp-1][0] == 0 and is_timestamp_during_a_seizure[timestamp][0] == 1):
            StartTime_and_EndTime_seizures.append(is_timestamp_during_a_seizure[timestamp][1])
    
    
    
    
    for i in range(len(StartTime_and_EndTime_seizures)-1):  # From StartTime_and_EndTime_seizures, we reason on the parity of the indices to know if we delimit a period of seizure or background to build is_period_a_seizure
        if i % 2 == 1:
            is_period_a_seizure.append([StartTime_and_EndTime_seizures[i], StartTime_and_EndTime_seizures[i+1], 1])
        if i % 2 == 0:
            is_period_a_seizure.append([StartTime_and_EndTime_seizures[i], StartTime_and_EndTime_seizures[i+1], 0])
    is_period_a_seizure.append([is_period_a_seizure[len(is_period_a_seizure)-1][1], is_timestamp_during_a_seizure[len(is_timestamp_during_a_seizure)-1][1], 0])
    
    
    
    background = []
    seizures = []

    fichier = {}  # a dictionary is created from is_period_a_seizure
    for i in range(len(is_period_a_seizure)):
        if is_period_a_seizure[i][2] == 0:
            background.append([is_period_a_seizure[i][0], is_period_a_seizure[i][1]])
        if is_period_a_seizure[i][2] == 1:
            seizures.append([is_period_a_seizure[i][0], is_period_a_seizure[i][1]])

            
            
    fichier["background"] = background
    fichier["seizures"] = seizures

    
    
    # we open the .json file which contains the output of the algorithm
    with open("output.json", "w") as outfile:
        json_object = json.dumps(fichier)
        outfile.write(json_object)
    return background,seizures

print(jeppesen("C:/Users/thibault/Documents/CS 1A/Pôle projet IA/Github P25/Projet-Epilepsie/database_processed/train/00001006/rr_00001006_s001_t001.csv","C:/Users/thibault/Documents/CS 1A/Pôle projet IA/Github P25/Projet-Epilepsie/database_processed/train/00001006/00001006_s001_t001.json",'Modified_csi*Slope'))

def convert_args_to_dict(args: argparse.Namespace) -> dict:
    """
    Convert argparse arguments into a dictionnary.

    From an argparse Namespace, create a dictionnary with only inputed CLI
    arguments. Allows to use argparse with default values in functions.

    parameters
    ----------
    args : argparse.Namespace
        Arguments to parse

    returns
    -------
    args_dict : dict
        Dictionnary with only inputed arguments
    """
    args_dict = {
        argument[0]: argument[1]
        for argument
        in args._get_kwargs()
        if argument[1] is not None}

    return args_dict


def parse_evaluate_args(
        args_to_parse) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--rr-path',
                        dest='enregistrement_path',
                        required=True)
    parser.add_argument('--annotation-path',
                        dest='json_path',
                        required=True)
    parser.add_argument('--jeppesen-feature-name',
                        dest='JEPPESEN_FEATURE_NAME',
                        required=True)
    parser.add_argument('--sliding-window',
                        dest='sliding_window')
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":
    args = parse_evaluate_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    jeppesen(**args_dict)
