"""
This script is used to detect seizures by using a SVM

INPUT : .csv contenant les intervalles RR
OUTPUT : fichier json avec les crises et leurs début de crise

"""

from sklearn import model_selection
import joblib
import os
import json
import sys
import numpy as np
import csv
import scipy.stats as scp
import statistics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
import argparse


#path = sys.argv[1]


def convert(fichierCSV):  # plus ou moins vérifiée et fonctionelle
    """
    converts CSV file in bpm array

    In the array the array there is : 
    array[i]= [temps, bpm, deltat]
    """
    with open(fichierCSV) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        tab = [row for row in readCSV]
    res = []
    for i in range(1, len(tab)):
        if i == 1:
            res.append([0, (60*1000)/float(tab[i][2]), float(tab[i][2])])

        else:
            res.append(GET_TIME_BPM_DELTAT(tab,res,i))

    return np.array(res)

def GET_TIME_BPM_DELTAT(tab,res,i):
    return [res[-1][0]+float(tab[i][2]), (60*1000) /float(tab[i][2]), float(tab[i][2])]

#ECG = convert(path)
# print(ECG)
# plt.plot(ECG[:,0],ECG[:,1])


def flatten(array):  
    """
    flattens the array with mean over 15 HR 
    array[i]=moy[i-7,i+8]
    """
    res = []
    for i in range(len(array)):
        if i >= 7 and i <= len(array)-8:
            res.append([array[i][0], np.median(
                (array[i-7:i+8, 1])), array[i][2]])
    return np.array(res)

#flattened=flatten(ECG)
# print(flattened)


def calculate_gradient(array):
    """
    input : flattened array [temps,bpm,deltat]
    output : The  gradient∇HRis  computed  as  the  gradient  of  the  linear  fit on  10  HR  measurements (cf Article)
    """
    res = []
    for i in range(len(array)):
        if i >= 9:
            res.append([array[i][0], scp.linregress(
                array[i-9:i+1, 0:2]).slope*1000])
    return np.array(res)

'''
gradient=calculate_gradient(flattened)
plt.plot(gradient[:,0]/1000,gradient[:,1])
cropped_array=ECG[len(ECG)-len(gradient):,:]
cropped_flattened=flattened[len(flattened)-len(gradient):,:]
plt.plot(cropped_array[:,0]/1000,cropped_array[:,1])
plt.plot(cropped_flattened[:,0]/1000,cropped_flattened[:,1])
plt.show()'''


def calculate_HRbase(array, t_start):
    """
    input : array [temps, bpm]
    output : HRbase ( mean bpm over 60 secs before t_start)

    
    """
    L = []
    if t_start > 60*1000:  # attention array a le temps en millisecondes
        for j in range(len(array)):
            if array[j][0] > t_start - 60*1000:
                L.append(array[j][1])

    else:
        for j in range(len(array)):
            L.append(array[j][1])

    return statistics.mean(L)


def calculate_HRrest(array):
    """
    Article pas très clair, on abandonne l'idée de mettre HRrest en feature parce que ça pose problème si les conditions de
    l'article ne sont pas vérifiées (on aurait jamais d'actualisation de HRrest en HRbase).
    """

    HRbase = calculate_HRbase(array)


def calculate_sdsd(gradient, indice_debut, indice_fin):
    gradient_crop = []
    for i in range(indice_debut, indice_fin+1):
        gradient_crop.append(gradient[i][1])
    gradient_crop = np.array(gradient_crop)
    return statistics.stdev(gradient_crop)


def hri_extract(array,total=False):
    """
    input : array [temps, bpm]
    output : [t_start,t_end,delta_t,deltaHR,HR_peak,HR_base,HR_start,sdsd] si total = True
             [HRbase,HRpeak,HRstart,sdsd] si total = False

    algo en O(n²) n=len(array[0]), il est possible de le coder en O(n). Les performances sont acceptables cependant.
    """

    # flatten the array
    array = flatten(array)

    # Compute HR data using gradient Hk
    gradient = calculate_gradient(array)
    #Si l'enregistrement n'est pas assez long
    if array.size == 0:
        return []
    #faire en sorte que array soit de la meme taille que gradient
    cropped_array = array[len(array)-len(gradient):, :]
    array = cropped_array
    # gradient est de taille plus petite que array et ne commencent pas en meme temps
    # solution : cut les premiers elements de array
    L = []
    i = 0
    while i <= len(gradient)-1:
        if gradient[i][1] > 1:

            indice_debut = i  # debut de HRI

            for j in range(indice_debut, len(array)):
                if gradient[j][1] < 0:
                    indice_fin = j
                    break
                elif j == len(array)-1:
                    indice_fin = j

            # Look back for last deltaHk < 0 : Start of HRI tstart
            for j in range(1, i):
                if gradient[j][1] <= 0.2:
                    indice_debut = j

            t_start = array[indice_debut][0]

            t_end = array[indice_fin][0]
            delta_HR = array[indice_fin][1] - array[indice_debut][1]

            #print(t_start,indice_debut,indice_fin,delta_HR,delta_HR/(t_end-t_start))

            #if delta_HR > 10 and delta_HR*1000/(t_end-t_start) > 0.35 and ... modified in order to better detect crisis
            if delta_HR > 7 and delta_HR*1000/(t_end-t_start) > 0.25 and array[indice_fin][1]/calculate_HRbase(array,t_start)>1.1:

            #Significant HRI detected
                if total==False:
                    L.append([calculate_HRbase(array,t_start),array[indice_fin][1],array[indice_debut][1],calculate_sdsd(gradient,indice_debut,indice_fin)])
                else:
                    L.append([t_start, t_end, t_end-t_start, delta_HR, array[indice_fin][1],calculate_HRbase(array,t_start) ,array[indice_debut][1],calculate_sdsd(gradient,indice_debut,indice_fin)])
            i=indice_fin+1
        i+=1
        
    return L
            #check ou appel fonction delta_HRrest
#features=hri_extract(ECG)[0]

    # check ou appel fonction delta_HRrest
#features = hri_extract(ECG)[0]

# L.append([features])

def calculate_F_beta(beta, Se, PPV):
    """
    input : beta (facteur qui donne plus ou moins d'importance à la sensitivité),
            Se (Sensitivité)
            PPV : Predictive value of the support vector machine (SVM)

    """

    F_beta = (1 + beta**2) * (Se * PPV) / ((beta**2) * Se + PPV)
    return F_beta

def feature_selection(features_pool, patients_pool):
    """
    input= a set of features that could be selected (features_pool)
           a set of patients  

    calculates F_b score for each feature

    On initialise feat_sel (les features sélectionnées) à 0, et le score à 0.
    """
    feat_sel = set()
    Fb_prev = 0
    Fb_new = 0
    while Fb_prev <= Fb_new:
        feat_pool_test = features_pool.difference(feat_sel)
        for feat in feat_pool_test:
            feat_test = set()
            feat_test.update(feat_sel)
            feat_test.add(feat)


def evaluate_F5_score(alpha):
    """
    A relier avec la partie Feature_selection
    Mise de côté pour l'instant car les meilleures features ont déjà été sélectionnées par l'article.
    """


def create_svm():
    """
    Fonction qui crée le SVM
    """
    classifier = svm.SVC(kernel="rbf", class_weight="balanced",gamma=0.004281332398719396, C=3.593813663804626)
    return classifier





# On choisit 10 valeurs pour C, entre 1e-2 et 1e3
C_range = np.logspace(-1, 1, 10)

# On choisit 10 valeurs pour gamma, entre 1e-2 et 10
gamma_range = np.logspace(-3, -1, 20)

# grille de paramètres
param_grid = {'C': C_range, 'gamma': gamma_range}


def optimize_hyperparameters(X_train_std, y_train, param_grid):
    """
    Input : X_train_std, y_train
    Output : Prints optimal svm parameters
    """
    # Critère de sélection du meilleur modèle
    score = 'roc_auc'

    grid = model_selection.GridSearchCV(svm.SVC(kernel='rbf'),
                                        param_grid,
                                        cv=5,  # 5 folds de validation croisée
                                        scoring=score,
                                        verbose=3)

    # faire tourner la recherche sur grille
    grid.fit(X_train_std, y_train)

    print("The optimal parameters are {} with a score of {:.2f}".format(grid.best_params_, grid.best_score_))


def classifier_manuel(HRI_features, jsonfile):
    """
    INPUT : (feature_list,jsonpath)
    OUTPUT : return 0 if HRI_features correspond to background
    1 if seizure
    """
    marge_debut = 10  # 10secondes
    marge_fin=30
    with open(jsonfile) as json_data:
        data_dict = json.load(json_data)
    t_start = HRI_features[0]
    t_end = HRI_features[1]

    seizs_intervals = data_dict["seizure"]
    if seizs_intervals == []:
        return (0,-1)
    for i,interval in enumerate(seizs_intervals):
        if t_start/1000 >= interval[0]-marge_debut and t_end/1000 <= interval[1]+marge_fin:
            #print('pos')
            return (1,i)
        else:
            return (0,i)

#print(hri_extract(convert(path),True))
#print(classifier_manuel(feats,'C:/Users/Maxime/Desktop/Pole_IA_Epilepsie/Projet-Epilepsie/database_processed/dev/00008544/00008544_s004_t008.json'))


def get_real_hri(array):
    pass

def annot_bdd(csv, jsonfile):
    '''
    INPUT : (csvpath,jsonpath)
    OUTPUT : (array_of_feature_list,array_of_0s_and1s) 1 indicating that feature_list corresponds to a seizure
    '''
    seizs = []
    list_csv=convert(csv)
    features_list_all = hri_extract(list_csv,True)
    features_list = GET_FOUR_FEATS(features_list_all)
    for features in features_list_all:
        seizs.append(classifier_manuel(features, jsonfile))

    # print(len(features_list),len(seizs))
    


    return features_list, true_seizs(features_list,seizs)


def true_seizs(features_list,seizs):
    '''

    '''
    res=[0 for _ in range(len(seizs))]
    for i in range(len(seizs)):
        val,interv=seizs[i]
        if val==1:
            for j in range(len(seizs)):
                val2,interv2=seizs[j]
                if interv2==interv and val2==1 and j!=i:
                    if features_list[i][2]-features_list[i][1]<features_list[j][2]-features_list[j][1]:
                        break
                    
                    res[i]=1
    return res



def create_labeled_db(dbpath):
    """
    INPUT : dataset path
    OUTPUT : x_train,y_train where x_train isn't normalized
    """
    bigfeatures_list = []
    bigseiz_list = []
    index=0
    for patient in os.scandir(dbpath):
        index+=1
        patient_d = os.path.join(dbpath, patient)
        #print(patient)
        for csvfile in os.scandir(patient_d):
           _,extension= os.path.splitext(csvfile)
           if extension=='.csv':
            csv_path=csvfile.path
            json_path=get_json_path_from_csv_file_and_patient(patient_d,csvfile)
            feats,seizs=annot_bdd(csv_path,json_path)
            for i in range(len(seizs)):
                bigfeatures_list.append(feats[i])
                bigseiz_list.append(seizs[i])
        progress(index,526)
    return bigfeatures_list,bigseiz_list


def get_json_path_from_csv_file_and_patient(patientdir,csvfile):
    return os.path.join(patientdir,csvfile.name[3:len(csvfile.name)-4]+".json")




def normalize_training_set(x):
    '''
    input: array of feature list 
    eg : [feature_list1,feature_list2] where feature_list1 = [feat1,feat2,feat3,feat4]
    output : return same shape array but normalized (all data between -1 and 1) and scaler
    '''
    std_scale=preprocessing.StandardScaler().fit(x)
    print("scaler : " , std_scale)
    return std_scale.transform(x),std_scale



def train_svm(classifier, x, y):
    """
    input : classifier -> svm , x -> list de features  , y -> list de 0 ou 1 correspondant à crises ou non
    output : (None) ; trains svm 
    """
    print(x,y)

    classifier.fit(x, y)



def predict(classifier, x):
    '''
    INPUT : (model,normalized_array_of_feature_list)
    OUTPUT : array of 0 and 1s with 1s indicating that features are indicative of seizure
    '''
    return classifier.predict(x)


def predict_ecg(classifier, std_scale, ecg):
    '''
    INPUT : (model,scaler,ecg array) ecg array = [date,bpm,temps]
    OUTPUT : an array containing the times of seizure starts.
    eg : output = [] -> no seizure detected
         output = [122000,200000] -> seizures starts at 122000 ms and 200000 ms
    '''
    features = hri_extract(ecg)
    features_tot=hri_extract(ecg,True)
    #print(features_tot)
    if len(features)!=0:
        features_scaled=std_scale.transform(features)
    
        prediction=classifier.predict(features_scaled)
    else:
        return []
    times=[]
    for i in range(len(prediction)):
        if prediction[i] == 1:
            times.append(features_tot[i][0])
    return times


def load_model():
    classifier = joblib.load('models/model_centralesupelec.joblib')
    return classifier

def load_scaler():
    scaler=joblib.load('models/scaler_centralesupelec.joblib')
    return scaler

def save_model():
    joblib.dump(classifier,"models/model_centralesupelec.joblib")

def save_scaler():
    joblib.dump(std_scale,"models/scaler_centralesupelec.joblib")


#The optimal parameters are {'C': 3.593813663804626, 'gamma': 0.004281332398719396} with a score of 0.70

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def GET_FOUR_FEATS(array):
    res=[]
    for feats in array:
        res.append([feats[5],feats[4],feats[6],feats[7]])
    return res

def vandercasteele(csv_path):
    '''
    INPUT: csv_path
    OUTPUT: write in jsonfile 'output' an array containing the times of seizure starts.
    eg : output = [] -> no seizure detected
         output = [122000,200000] -> seizures starts at 122000 ms and 200000 ms 

    '''
    classifier=load_model()
    std_scale=load_scaler()
    dico={}
    dico["seizures"]=predict_ecg(classifier,std_scale,convert(csv_path))




    with open("src/projet_CS/detection_algorithms/output.json","w") as outfile:
        json_object = json.dumps(dico)
        outfile.write(json_object)

#vandercasteele(sys.argv[1])
#DB_PATH_TRAIN='C:/Users/Maxime/Desktop/Pole_IA_Epilepsie/Projet-Epilepsie/database_processed_clean'
def full_code_to_train_svm(DB_PATH_TRAIN):

    global classifier
    global std_scale
    classifier = create_svm()
    f,s=create_labeled_db(DB_PATH_TRAIN)
    #print(np.sum(s))
    x_train,std_scale=normalize_training_set(f)

    y_train = s
    train_svm(classifier,x_train,s)
    #optimize_hyperparameters(x_train, y_train, param_grid)
    save_model()
    save_scaler()
    return classifier,std_scale

#full_code_to_train_svm(DB_PATH)

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
    parser.add_argument('--csv_path',
                        dest='csv_path',
                        required=False)
    parser.add_argument('--db_train_path',
                        dest='db_train_path',
                        required=False)
    
    args = parser.parse_args(args_to_parse)

    return args



if __name__ == "__main__":
    args = parse_evaluate_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    
    if('db_train_path' in args_dict):
        full_code_to_train_svm(args_dict['db_train_path'])
    else:
        vandercasteele(args_dict['csv_path'])