from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import json
import os
from detection_algorithms.fisher_clean import fisher
import time
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import detection_algorithms.vandercasteele_clean as vandercasteele
import joblib
import sys
import argparse
from detection_algorithms.Jeppesen.Jeppesen import jeppesen


OUTPUT_PATH='./src/projet_CS/detection_algorithms/output.json'
VALIDITY_TIME_OF_DETECTION_BEFORE_SEIZURE=45 #in seconds
JEPPESEN_FEATURE_NUMBER_DEFAULT=30
db_path='../database_processed_clean/'


def evaluation(db_path,algorithm,jeppesen_feature_number=JEPPESEN_FEATURE_NUMBER_DEFAULT):
    """
    Input : 
        db_path=path of the part of the database which will be evaluated. The samples need to be sorted by patient, in folders with a unique patient id. The annotations must be put in the same folder
        algorithm= 'fisher', 'jeppesen' or 'vandercasteele 
    Output :

    """
    (dataset_annotation_vector,dataset_prediction_vector)=dataset_evaluate(db_path,algorithm,jeppesen_feature_number) 
    dataset_annotation_vector_np=np.array(dataset_annotation_vector)
    #np.save('../../output/models_evaluation/annot_vdc.npy',dataset_annotation_vector_np)
    dataset_prediction_vector_np=np.array(dataset_prediction_vector)
    #np.save('../../output/models_evaluation/pred_vdc.npy',dataset_prediction_vector_np)
   
    print('F1 score is ',f1_score(dataset_annotation_vector,dataset_prediction_vector))             #F1 = 2 * (precision * recall) / (precision + recall)
    print('Accuracy is ',accuracy_score(dataset_annotation_vector,dataset_prediction_vector))       #In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print('Recall is ',recall_score(dataset_annotation_vector,dataset_prediction_vector))         #The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
    print('Precision is ',precision_score(dataset_annotation_vector,dataset_prediction_vector))   #The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    
    array=confusion_matrix(dataset_annotation_vector,dataset_prediction_vector)
    #np.save('../../output/models_evaluation/confusion_matrix_feature30_filter.npy',array)
    df_cm = pd.DataFrame(array, index = ['Pas de crise','Crise'], columns = ['Pas de crise','Crise'])
    sn.heatmap(df_cm, cmap=sn.cm.rocket_r, annot=True,vmin=0)
    plt.show()



def dataset_evaluate(db_path,algorithm,jeppesen_feature_number):
    """
    Input :
        db_path=path of the part of the database which will be evaluated. The samples need to be sorted by patient, in folders with a unique patient id. The annotations must be put in the same folder
        algorithm= 'fisher', 'jeppesen' or 'vandercasteele 
    Output :
        Two vectors with 0 and 1 describing the evaluation of the predictions of the algorithm
    """
    dataset_annotation_vector=[]
    dataset_prediction_vector=[]
    patient_list=os.listdir(db_path)
    for patient in patient_list:
        file_names=os.listdir(db_path+patient)
        for file_parsing in file_names:
            file=db_path+patient+'/'+file_parsing
            type=file_parsing.split('.')[1]
            if type=='json' and 'rr_'+file_parsing.split('.')[0]+".csv" in file_names:
                rr_path=db_path+patient+'/'+'rr_'+file_parsing.split('.')[0]+".csv"
                annotat_path=file
                print(rr_path)
                try:
                    sample_results_list=evaluation_one_sample(algorithm,rr_path,annotat_path,jeppesen_feature_number)
                    for el in sample_results_list:
                        dataset_annotation_vector.append(el[0])
                        dataset_prediction_vector.append(el[1])
                except:
                    print('une erreur a été détectée')
                """
                sample_results_list=evaluation_one_sample(algorithm,rr_path,annotat_path,jeppesen_feature_number)
                for el in sample_results_list:
                    dataset_annotation_vector.append(el[0])
                    dataset_prediction_vector.append(el[1])
                """
    return(dataset_annotation_vector,dataset_prediction_vector)


def evaluation_one_sample(algorithm, csv_path,annotation_path,jeppesen_feature_number):
    """
    Input : 
            algorithm = name of the algorithm used for seizure detection
                        can take values 'fisher', 'jeppesen', 'vandercasteele'
            csv_path = path of the csv file of the sample
            annotation_path = path of the json file with the annotation
    
    Output : 
            sample_results_list =a list of lists with the format [a,b] with a=1 if there is a seizure in original annotation path, a=0 otherwise
                                                                            b=1 if a seizure is well detected by the algorithm, b=0 otherwise
            
                    This list is used to evaluate if the considered detection algorithm detected well each seizure of the given sample
                    There will be at least one element [a,b] for each seizure (annotated or detected by the algorithm) in sample_results_list. It evaluates the prediction
                    of the algorithm : [0,1] means a false positive, [1,0] means that a seizure annotated hasn't been detected.
    
    Conditions for a true detection :

    """

    if algorithm=='fisher':
        sample_results_list=evaluation_one_sample_fisher(csv_path,annotation_path)
        print(sample_results_list)
    if algorithm=='jeppesen':
        sample_results_list=evaluation_one_sample_jeppesen(csv_path,annotation_path,jeppesen_feature_number)
    if algorithm=='vandercasteele':
        sample_results_list=evaluation_one_sample_vandercasteele(csv_path,annotation_path)
    return(sample_results_list)


def evaluation_one_sample_jeppesen(csv_path,annotation_path,jeppesen_feature_number):
    """
    Input : 
        csv_path = path of the csv file of the record
        annotation_path = path of the json file with the annotation
        
    Output : 
            sample_results_list =a list of lists with the format [a,b] with a=1 if there is a seizure in original annotation path, a=0 otherwise
                                                                            b=1 if a seizure is well detected by the algorithm, b=0 otherwise
            
                    This list is used to evaluate if the considered detection algorithm detected well each seizure of the given sample
                    There will be at least one element [a,b] for each seizure (annotated or detected by the algorithm) in sample_results_list. It evaluates the prediction
                    of the algorithm : [0,1] means a false positive, [1,0] means that a seizure annotated hasn't been detected.
    
    Particularities of Jeppesen that impact the evaluation methods :
        - We suppose that Jeppesen works in an "offline" mode and that it gives for each sample an output with the same format as the annotation_path. 

    Conditions for a good detection :
            a seizure is considered as well detected if there is a predicted seizure which starts within the range [t_start_annotated - VALIDITY_TIME ; t_end_annotated]
    """
    jeppesen(csv_path,annotation_path,jeppesen_feature_number)
    time.sleep(2)
    output_evaluation=[]
    annotation=open_annotation(annotation_path)
    json_output=open_annotation(OUTPUT_PATH)
    if len(annotation["seizure"])==0:
        output_evaluation=no_annotated_seizure_evaluation(output_evaluation,json_output)
    elif len(annotation["seizure"])==1:
        output_evaluation=one_annotated_seizure_evaluation(output_evaluation,json_output,annotation)
    else:
        output_evaluation=multiple_annotated_seizures_evaluation(output_evaluation,json_output,annotation)
    
def evaluation_one_sample_vandercasteele(csv_path,annotation_path):
    """
    Input : 
        csv_path = path of the csv file of the record
        annotation_path = path of the json file with the annotation
        
    Output : 
            sample_results_list =a list of lists with the format [a,b] with a=1 if there is a seizure in original annotation path, a=0 otherwise
                                                                            b=1 if a seizure is well detected by the algorithm, b=0 otherwise
            
                    This list is used to evaluate if the considered detection algorithm detected well each seizure of the given sample
                    There will be at least one element [a,b] for each seizure (annotated or detected by the algorithm) in sample_results_list. It evaluates the prediction
                    of the algorithm : [0,1] means a false positive, [1,0] means that a seizure annotated hasn't been detected.
    
    Particularities of Vandercasteele that impact the evaluation methods :
        - We suppose that Vandercasteele works in an "offline" mode and that it gives for each sample an output with the same format as the annotation_path. 

    Conditions for a good detection :
            a seizure is considered as well detected if there is a predicted seizure which starts within the range [t_start_annotated - VALIDITY_TIME ; t_end_annotated]
    """

    output_evaluation=[]
    
    res=vandercasteele.annot_bdd(csv_path,annotation_path)  
    time.sleep(2)
    feats=res[0]
    annot=res[1]
    std_scale=joblib.load('models/scaler_centralesupelec.joblib')
    classifier=joblib.load('models/model_centralesupelec.joblib')
    if len(feats)==0: # il faut voir avec le fichier json
        json_output=open_annotation(annotation_path)
        
        if len(json_output["seizure"])==0:
            output_evaluation.append([0,0])
        else:
            for _ in range(len(json_output["seizure"])):
                output_evaluation.append([1,0])
    else:
        features_scaled=std_scale.transform(feats)
        predicts=classifier.predict(features_scaled)
        for i in range(len(annot)):
            output_evaluation.append([annot[i],predicts[i]])
    return output_evaluation


def evaluation_one_sample_fisher(csv_path,annotation_path):
    """
    Input : 
        csv_path = path of the csv file of the record
        annotation_path = path of the json file with the annotation
        
    Output : 
            sample_results_list =a list of lists with the format [a,b] with a=1 if there is a seizure in original annotation path, a=0 otherwise
                                                                            b=1 if a seizure is well detected by the algorithm, b=0 otherwise
            
                    This list is used to evaluate if the considered detection algorithm detected well each seizure of the given sample
                    There will be at least one element [a,b] for each seizure (annotated or detected by the algorithm) in sample_results_list. It evaluates the prediction
                    of the algorithm : [0,1] means a false positive, [1,0] means that a seizure annotated hasn't been detected.
    
    Particularities of fisher that impact the evaluation methods :
        - We suppose that Fisher works in an "offline" mode and that it gives for each sample an output with the same format as the annotation_path. 
        - Fisher needs to calculate HR on a LONG_WINDOW, that has a default length of 5 minutes. Most samples of TUH database have seizures within the 5 first minutes of recording, 
        so we decided to make at the beginning of the sample an "evolving" LONG_WINDOW calculating the mean HR on the first minutes of recording. However, we consider that we still need 
        at least 1min30s of background before seizures at the beginning of the samples.
        Beyond 5 minutes of recording, the length of LONG_WINDOW is set to 5 minutes.  However, as the algorithm's goal is to detect well a seizure appearing during a background period, 
        we considered that (in the context of the performances evaluation of the detection algorithm), the detection of a seizure of a certain sample should not be influenced by the
        seizures that happened before. For instance, if two seizures are only separated with 2min, the LONG_WINDOW (length 5min) used to detect the second seizure will contain the first seizure, and so
        the presence of the first seizure could increase the mean HR calculated on the LONG_WINDOW, and the second seizure would not be detected, while if there was only background 5 minutes before the 
        seizure, it could be well detected.

        So we decided to artificially "split" the samples with several seizures that are not separated by at least 5 minutes.
    

    Remarks:
            /!\ compare the results with ~30 prcent of people on which such method can work 
            It would be interesting to evaluate this algorithm on a dataset with only patients with ictal tachychardia


    Conditions for a good detection :
            a seizure is considered as well detected if there is a predicted seizure which starts within the range [t_start_annotated - VALIDITY_TIME ; t_end_annotated]
    """
    output_evaluation=[]
    annotation=open_annotation(annotation_path)
    json_output=open_annotation(OUTPUT_PATH)
    if len(annotation["seizure"])==0:
        fisher(csv_path)
        time.sleep(1)
        output_evaluation=no_annotated_seizure_evaluation(output_evaluation,json_output)
    elif len(annotation["seizure"])==1:
        fisher(csv_path)
        time.sleep(1)
        output_evaluation=one_annotated_seizure_evaluation(output_evaluation,json_output,annotation)
    else:
        output_evaluation=fisher_multiple_annotated_seizures_evaluation(output_evaluation,csv_path,annotation)

def multiple_annotated_seizures_evaluation(output_evaluation,json_output,annotation):
    if len(json_output["seizure"])==0:
        for j in range(len(annotation)):          #Aucune des multiples crises annotées n'a été détectée
            output_evaluation.append([1,0])
    else :
        for seizure_number in range(len(annotation["seizure"])): #Un traitement par crise annotée
            output_evaluation,json_output=evaluation_on_one_of_the_multiple_annotated_seizures(output_evaluation,json_output,annotation,seizure_number)
        if len(json_output["seizure"])>0:                              #Les crises restantes sont des mauvaises détections/fausses alertes
            for n in range(len(json_output["seizure"])):
                output_evaluation.append([0,1])
    return(output_evaluation)

def fisher_multiple_annotated_seizures_evaluation(output_evaluation,csv_path,annotation):
    for seizure_number in range(len(annotation["seizure"])):    #Un traitement par crise va être effectué
        if seizure_number==0:
            output_evaluation=fisher_evaluation_on_the_first_of_the_multiple_annotated_seizures(output_evaluation,csv_path,annotation)
        else:
            output_evaluation=fisher_evaluation_on_the_following_multiple_annotated_seizures(output_evaluation,csv_path,annotation)
    return(output_evaluation)

def fisher_evaluation_on_the_first_of_the_multiple_annotated_seizures(output_evaluation,csv_path,annotation):
    if annotation["seizure"][0][0]>=90: #Première crise apparait au delà d'1min30s, on prend en compte
        new_annotation={}
        fisher(csv_path,sample_end=annotation["seizure"][1])
        time.sleep(2)
        json_output=open_annotation(OUTPUT_PATH)
        new_annotation["seizure"]=[]
        new_annotation["seizure"].append(annotation["seizure"][0])
        new_annotation["background"]=[]
        new_annotation["background"].append(annotation["background"][0])
        res=one_annotated_seizure_evaluation(output_evaluation,json_output,new_annotation)
        for el in res:
            output_evaluation.append(el)
    return(output_evaluation)

def fisher_evaluation_on_the_following_multiple_annotated_seizures(output_evaluation,csv_path,annotation,seizure_number):
    t_end=annotation["seizure"][seizure_number][1]
    if annotation["seizure"][seizure_number][0]-annotation["seizure"][seizure_number-1][1]>90:   #Ecart de plus d'1min30s entre la fin de la crise précédente et le début de la nouvelle, sinon on ne fait rien
        new_annotation={}
        if seizure_number==len(annotation["seizure"])-1:        #Si c'est la dernière crise, le traitement est particulier car un background supplémentaire apparait à la fin
            fisher(csv_path,sample_beginning=annotation["seizure"][seizure_number-1][1]+20)
            time.sleep(2)
            json_output=open_annotation(OUTPUT_PATH)
            new_annotation["seizure"]=[]
            new_annotation["seizure"].append(annotation["seizure"][seizure_number])
            new_annotation["background"]=[]
            new_annotation["background"].append(annotation["background"][seizure_number])
            if not annotation["background"][-1] in new_annotation["background"]:
                new_annotation["background"].append(annotation["background"][-1])
        else:                                       #Sinon, on aura un background puis une crise
            fisher(csv_path,sample_beginning=annotation["seizure"][seizure_number-1][1]+20,sample_end=t_end)   #On laisse un écart
            time.sleep(2)
            new_annotation["seizure"]=[]
            new_annotation["seizure"].append(annotation["seizure"][seizure_number])
            new_annotation["background"]=[]
            new_annotation["background"].append([annotation["background"][seizure_number][0]+20,annotation["background"][seizure_number][1]])
                                #On traite les résultats comme si une seule crise était considérée
            res=one_annotated_seizure_evaluation(output_evaluation,json_output,new_annotation)
            for el in res:
                output_evaluation.append(el)
    return(output_evaluation)

def evaluation_on_one_of_the_multiple_annotated_seizures(output_evaluation,json_output,annotation,seizure_number):
    annotated_t_start=annotation["seizure"][seizure_number][0]
    annotated_t_end=annotation["seizure"][seizure_number][1]
    good_predictions=[]
    detection=False
    for detection_number in range(len(json_output["seizure"])):
        prediction_t_start=json_output["seizure"][detection_number][0]
        if seizure_good_detection_assertion(annotated_t_start,annotated_t_end,prediction_t_start):
            good_predictions.append(detection_number)
            output_evaluation.append([1,1])
            detection=True
    if not detection:
        output_evaluation.append([1,0])
    for i in sorted(good_predictions, reverse = True):
        del(json_output["seizure"][i])
    return(output_evaluation,json_output)


def no_annotated_seizure_evaluation(output_evaluation,json_output):
    if len(json_output["seizure"])>0:
        for detected_seizure in range(len(json_output["seizure"])):    
            output_evaluation.append([0,1])               #False positives
    else:
        output_evaluation.append([0,0])                
    return(output_evaluation)

def one_annotated_seizure_evaluation(output_evaluation,json_output,annotation):
    annotated_t_start=annotation["seizure"][0][0]
    annotated_t_end=annotation["seizure"][0][1]

    if len(json_output["seizure"])==0:   #Aucune crise n'a été détectée
        output_evaluation.append([1,0])
    elif len(json_output["seizure"])==1:  #Une crise a été détectée
        predicted_t_start=json_output["seizure"][0][1]
        if seizure_good_detection_assertion(annotated_t_start,annotated_t_end,json_output[0][1]):  #Bonne détection de la crise
            output_evaluation.append([1,1])
        else:                                                              #Mauvaise détection de la crise
            output_evaluation.append([1,0])
    else:                                   #Multiple détection de crises alors qu'une seule est annotée : un traitement spécifique à chaque crise détectée va devoir être effectué
        output_evaluation=one_annotated_seizure_but_multiple_detections_evaluation(output_evaluation,json_output,annotation)
    return output_evaluation

def one_annotated_seizure_but_multiple_detections_evaluation(output_evaluation,json_output,annotation):
    seizure_detected=False
    for detected_seizure in json_output["seizure"]:
        if not seizure_good_detection_assertion(annotation[0][0],annotation[0][1],detected_seizure[0]):   #Cette détection n'a rien à voir avec la seule crise annotée présente dans l'enregistrement : c'est une fausse alerte
            output_evaluation.append([0,1])
        elif seizure_good_detection_assertion(annotation[0][0],annotation[0][1],detected_seizure[0]) and seizure_detected==False:  #La crise est bien détectée, et pour la première fois
            seizure_detected==True
            output_evaluation.append([1,1])
        elif seizure_good_detection_assertion(annotation[0][0],annotation[0][1],detected_seizure[0]) and seizure_detected==True:
            #On considère que c'est la suite de la détection de la crise, donc on ne fait rien
            '''do nothing'''
        else:
            output_evaluation.append([0,1])
    if seizure_detected==False:                 #La crise n'a pas été détectée 
        output_evaluation.append([1,0])  
    return(output_evaluation)


def seizure_good_detection_assertion(annotated_t_start, annotated_t_end, predicted_t_start):
    """
    Is used to evaluate if a prediction about an annotated seizure can be considered as accurate
    """
    if (predicted_t_start>=annotated_t_start-VALIDITY_TIME_OF_DETECTION_BEFORE_SEIZURE and predicted_t_start<=annotated_t_end):
        return True
    else:
        return False




def open_annotation(annotation_path): 
    """
    input : path of the .json annotation file
    output : the content of the annotation file (a dictionary)
    """
    with open(annotation_path) as json_annotation:
        annotation=json.load(json_annotation)
        return(annotation)


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
    parser.add_argument('--db-path',
                        dest='db_path',
                        required=True)
    parser.add_argument('--algorithm',
                        dest='algorithm',
                        required=True)
    parser.add_argument('--jeppesen-feature-number',
                        dest='JEPPESEN_FEATURE_NUMBER')
    args = parser.parse_args(args_to_parse)

    return args

'''if __name__ == "__main__":
    args = parse_evaluate_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    evaluation(**args_dict)'''
evaluation(db_path,'vandercasteele')

