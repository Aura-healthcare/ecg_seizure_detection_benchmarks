import csv
import sys
import re
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse


DEFAULT_THRESHOLD = 1.3 #(30%)
DEFAULT_LARGE_WINDOW=5*60*1000 # temps en ms
DEFAULT_SHORT_WINDOW=10000 #temps en ms
START_TIME=20
DEFAULT_SAMPLE_BEGINNING=0
DEFAULT_SAMPLE_END=0
DISPLAY_GRAPH_BOOLEAN=True


def fisher(csv_path,threshold=DEFAULT_THRESHOLD,large_window=DEFAULT_LARGE_WINDOW,short_window=DEFAULT_SHORT_WINDOW,sample_beginning=DEFAULT_SAMPLE_BEGINNING,sample_end=DEFAULT_SAMPLE_END):
    """
    la fonction principale de l'algorithme Fisher
    Inputs :
        csv_path= the extracted content of the .csv file of the rr_intervals
        threshold= threshold used by the algorithm (1.3=30% ; 1.5=50%...)
        large_window= length of the large window
        short_window=
        sample_beginning= the starting time (in ms) of the cropped sample 
        sample_end= the ending time (in ms) of the cropped sample 
    
    The result will be written in the output.json file
    """
    (csv_list,final_short_and_large_windows_array,time_array)=fisher_calculation(csv_path,threshold,large_window,short_window,sample_beginning,sample_end)

    if DISPLAY_GRAPH_BOOLEAN:
        fisher_display(final_short_and_large_windows_array,time_array,threshold)

    fichier=json_writer(csv_list,time_array,threshold,sample_beginning)
    with open("src/projet_CS/detection_algorithms/output.json","w") as outfile:
        json_object = json.dumps(fichier)
        outfile.write(json_object)

def fisher_calculation(csv_path,threshold,large_window,short_window,sample_beginning,sample_end):

    csv_list=open_csv(csv_path)

    sample_end=sample_end*1000                  #in ms
    sample_beginning=sample_beginning*1000      #in ms

    csv_list=modify_the_length_of_the_sample(csv_list,sample_beginning,sample_end)

    csv_length=len(csv_list)

    long_window_hr_median_list=[]
    short_window_hr_median_list=[]
    time_array=[]
    date=[]

    for timestamp_treated in range(1,csv_length):
        current_timestamp=timestamp_treated
        heartbeat_number=0
        authorize_treatment_if_the_time_since_beginning_is_superior_to_start_time=True

        short_window_hr_median,current_timestamp,heartbeat_number,treatment_time=calculation_normal(csv_list,current_timestamp,large_window,short_window,heartbeat_number)
        
        if current_timestamp>=csv_length:
            break
        
        if authorize_treatment_if_the_time_since_beginning_is_superior_to_start_time:
            short_window_hr_median_list,long_window_hr_median_list,csv_list,time_array,date=actualize_arrays(short_window_hr_median_list,long_window_hr_median_list,short_window_hr_median,heartbeat_number,treatment_time,current_timestamp,timestamp_treated,csv_list,time_array,date)
            
    final_short_and_large_windows_array=[[long_window_hr_median_list[i],short_window_hr_median_list[i]] for i in range(len(short_window_hr_median_list))]
    return(csv_list,final_short_and_large_windows_array,time_array)

def actualize_arrays(short_window_hr_median_list,long_window_hr_median_list,short_window_hr_median,heartbeat_number,treatment_time,end_timestamp,beginning_timestamp,csv_list,time_array,date):
    short_window_hr_median_list.append(short_window_hr_median/heartbeat_number)
    long_window_hr_median_list.append(treatment_time/(end_timestamp-beginning_timestamp+1))
    csv_list=np.array(csv_list)
    time_array.append(np.sum(csv_list[1:end_timestamp,2:3].astype(np.float))/(1000*60))
    date.append(csv_list[end_timestamp][0])
    return(short_window_hr_median_list,long_window_hr_median_list,csv_list,time_array,date)

def calculation_normal(csv_list,current_timestamp,large_window,short_window,heartbeat_number):
    treatment_time_since_beginning=0
    short_window_hr_median=0
    while treatment_time_since_beginning<=large_window:   
        if current_timestamp>=len(csv_list):      #Si la longueur de l'enregistrement est inférieure à la fenêtre large
            break
        treatment_time_since_beginning+=float(csv_list[current_timestamp][2])
        if treatment_time_since_beginning >= large_window - short_window:    
            short_window_hr_median+=float(csv_list[current_timestamp][2])
            heartbeat_number+=1
        current_timestamp+=1
    return(short_window_hr_median,current_timestamp,heartbeat_number,treatment_time_since_beginning)  



def modify_the_length_of_the_sample(csv_list,sample_beginning,sample_end):
    """
    This function (optional) is used to modify the length of the sample, in the case the user wants to execute the algorithm on a part of the sample

    Input :
        csv_list = the extracted content of the .csv file of the rr_intervals
        sample_beginning = the starting time (in ms) of the cropped sample 
        sample_end = the ending time (in ms) of the cropped sample 
    
    Output :
        modified (cropped) csv list
    """
    if not sample_beginning==0:          #We want to work on a specific part of the sample that begins later than the original sample
        sample_beginning_timestamp=0
        total_time=0
        while total_time<sample_beginning:
            sample_beginning_timestamp+=1
            total_time+=float(csv_list[sample_beginning_timestamp][2])
    if not sample_end==0:                #We want to work on a specific part of the sample that ends sooner than the original sample
        sample_end_timestamp=0
        total_time=0
        while total_time<sample_end:
            sample_end_timestamp+=1
            total_time+=float(csv_list[sample_end_timestamp][2])
    if (not sample_beginning==0) and sample_end==0:  
        csv_list=csv_list[sample_beginning_timestamp:]
    elif sample_beginning==0 and (not sample_end==0):
        csv_list=csv_list[:sample_end_timestamp+1]
    elif (not sample_beginning==0) and (not sample_end==0):
        csv_list=csv_list[sample_beginning_timestamp:sample_end_timestamp+1]
    else:
        '''do nothing'''
    return(csv_list)

def fisher_display(classification_result,time_array,threshold):
    Y1=[classification_result[i][0] for i in range(len(classification_result))]
    Y2=[classification_result[i][1] for i in range(len(classification_result))]
    plt.plot(time_array,60*1000/np.array(Y1))
    plt.plot(time_array,60*1000*threshold/np.array(Y1))
    plt.plot(time_array,60*1000/np.array(Y2))
    plt.show()


def open_csv(csv_path):
    with open(csv_path) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        value_list=[row for row in readCSV]
        return(value_list)

def open_json(json_path): 
    """
    input : path of the .json annotation file
    output : the content of the annotation file (a dictionary)
    """
    with open(json_path) as json_annotation:
        annotation=json.load(json_annotation)
        return(annotation)

def json_writer(csv_list,time_array,threshold,sample_beginning):
    fichier={}
    long_window_hr_median_list=[csv_list[i][0] for i in range(len(csv_list))]
    short_window_hr_median_list=[csv_list[i][1] for i in range(len(csv_list))]
    seizures=[]
    background=[]
    en_background=True
    new_background=[sample_beginning+time_array[0]*60]      
    for i in range(len(long_window_hr_median_list)):
        if long_window_hr_median_list[i]/short_window_hr_median_list[i]>=threshold and en_background: #detection 1er 1
            new_background.append(sample_beginning+time_array[i]*60)
            background.append(new_background)
            en_background=False
            new_seizure=[sample_beginning+time_array[i]*60]
        elif long_window_hr_median_list[i]/short_window_hr_median_list[i]>=threshold : # en crise ( suite de 1)
            '''do nothing'''
    
        else:
            if not en_background: # sortie de crise
                new_seizure.append(sample_beginning+time_array[i]*60)
                seizures.append(new_seizure)
                en_background=True
                new_background=[sample_beginning+time_array[i]*60]
    new_background.append(sample_beginning+time_array[-1]*60)
    background.append(new_background)
    
    background_merged=[background[0]]                #On fusionne les détections de crises proches
    seizures_merged=[]

    if len(seizures)>0:
        seizures_merged.append(seizures[0])
        index_merged=[]
        for k in range(len(seizures[1:])):
            new_seizure=seizures[k+1]
            t_start=new_seizure[0]
            if t_start - seizures_merged[-1][1]<10:
                seizures_merged[-1][1]=new_seizure[1]
                index_merged.append([k,k+1])
            else:
                seizures_merged.append(seizures[k+1])
        for k in range(len(seizures_merged)-1):
            t_end=seizures_merged[k][1]
            following_t_start=seizures_merged[k+1][0]
            background_merged.append([t_end,following_t_start])
        background_merged.append(background[-1])
    fichier["background"]=background_merged
    fichier["seizure"]=seizures_merged
    return(fichier)


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
                        dest='rr_path',
                        required=True)
    parser.add_argument('--threshold',
                        dest='threshold')
    parser.add_argument('--large-window',
                        dest='large_window')
    parser.add_argument('--short-window',
                        dest='short_window')
    parser.add_argument('--sample-beginning',
                        dest='sample_beginning')
    parser.add_argument('--sample-end',
                        dest='sample_end')
    args = parser.parse_args(args_to_parse)

    return args


fisher('../database_processed_clean/00001006/rr_00001006_s001_t001.csv')

"""
if __name__ == "__main__":
    args = parse_evaluate_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    fisher(**args_dict)
"""