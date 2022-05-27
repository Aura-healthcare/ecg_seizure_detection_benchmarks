import csv
import sys
import re
import matplotlib.pyplot as plt
import json
import numpy as np


DEFAULT_THRESHOLD = 1.3 #(40%)
DEFAULT_LARGE_WINDOW=5*60*1000 # temps en ms
DEFAULT_SHORT_WINDOW=10000 #temps en ms
START_TIME=20
DEFAULT_SAMPLE_BEGINNING=0
DEFAULT_SAMPLE_END=0
#rr_path = sys.argv[1]
DISPLAY=False

def fisher_calculs(csv_path,threshold=DEFAULT_THRESHOLD,large_window=DEFAULT_LARGE_WINDOW,short_window=DEFAULT_SHORT_WINDOW,sample_beginning=DEFAULT_SAMPLE_BEGINNING,sample_end=DEFAULT_SAMPLE_END):

    with open(csv_path) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        tab=[row for row in readCSV]
    sample_end=sample_end*1000
    sample_beginning=sample_beginning*1000
    if not sample_beginning==0:
        sample_beginning_timestamp=0
        total_time=0
        while total_time<sample_beginning:
            sample_beginning_timestamp+=1
            total_time+=float(tab[sample_beginning_timestamp][2])
    #print(sample_beginning_timestamp)
    if not sample_end==0:
        sample_end_timestamp=0
        total_time=0
        while total_time<sample_end:
            sample_end_timestamp+=1
            total_time+=float(tab[sample_end_timestamp][2])
    #print(sample_end_timestamp)
    if (not sample_beginning==0) and sample_end==0:
        tab=tab[sample_beginning_timestamp:]
    elif sample_beginning==0 and (not sample_end==0):
        tab=tab[:sample_end_timestamp+1]
    elif (not sample_beginning==0) and (not sample_end==0):
        tab=tab[sample_beginning_timestamp:sample_end_timestamp+1]
    else:
        '''do nothing'''
    if (not sample_beginning==0) or (not sample_end==0):
        tablength=len(tab)
    else:
        tablength=len(tab)

    #print(tab[0])
    #print(tab[-1])
    tab_FCF=[]
    tab_FCPP=[]
    X=[]
    
    temps_arr=[]
    date=[]
    tempsdepuis=0   #en ms
    for timestamp in range(1,tablength):
        current_timestamp=timestamp #j=i
        tempsdepuis+=float(tab[timestamp][2])
        heartbeat_number=0
        temps_écoulé=0
        FCPP=0
        treatment=True
        
        if tempsdepuis<large_window and tempsdepuis>=START_TIME*1000:    #Période "intermédiaire" au début de l'algorithme
            while temps_écoulé<=tempsdepuis:
                if current_timestamp>=tablength:      #Si la longueur de l'enregistrement est inférieure à la fenêtre large
                    break
                temps_écoulé+=float(tab[current_timestamp][2])
                if temps_écoulé >= tempsdepuis - short_window:    
                    FCPP+=float(tab[current_timestamp][2])
                    heartbeat_number+=1
                current_timestamp+=1
        elif START_TIME*1000>tempsdepuis:
            treatment=False
        else:                                        
            while temps_écoulé<=large_window:         #Début des calculs
                if current_timestamp>=tablength:      #Si la longueur de l'enregistrement est inférieure à la fenêtre large
                    break
                temps_écoulé += float(tab[current_timestamp][2])
                if temps_écoulé >= large_window - short_window:
                    FCPP+=float(tab[current_timestamp][2])
                    heartbeat_number+=1
                current_timestamp+=1
        
        if current_timestamp>=tablength:
            break
        if treatment:
            tab_FCPP.append(FCPP/heartbeat_number)
            tab_FCF.append(temps_écoulé/(current_timestamp-timestamp+1))
            tab=np.array(tab)
            temps_arr.append(np.sum(tab[1:current_timestamp,2:3].astype(np.float))/(1000*60))
            date.append(tab[current_timestamp][0])
    tab_final=[[tab_FCF[i],tab_FCPP[i]] for i in range(len(tab_FCF))]
    return(tab_final,temps_arr)
    
def fisher_display(tabf,temps_arr,threshold):
    X=[i for i in range(len(tabf))]
    Y1=[tabf[i][0] for i in range(len(tabf))]
    Y2=[tabf[i][1] for i in range(len(tabf))]
    plt.plot(temps_arr,60*1000/np.array(Y1))
    plt.plot(temps_arr,60*1000*threshold/np.array(Y1))
    plt.plot(temps_arr,60*1000/np.array(Y2))
    plt.show()


def fisher(csv_path,threshold=DEFAULT_THRESHOLD,large_window=DEFAULT_LARGE_WINDOW,short_window=DEFAULT_SHORT_WINDOW,sample_beginning=DEFAULT_SAMPLE_BEGINNING,sample_end=DEFAULT_SAMPLE_END):
    (tab,temps_arr)=fisher_calculs(csv_path,threshold,large_window,short_window,sample_beginning,sample_end)
    #print(temps_arr[0])
    #print(temps_arr[-1])
    if DISPLAY:
        fisher_display(tab,temps_arr,threshold)
    fichier=json_writer(tab,temps_arr,threshold,sample_beginning)
    with open("src/projet_CS/detection_algorithms/output.json","w") as outfile:
        json_object = json.dumps(fichier)
        outfile.write(json_object)


def json_writer(tab,temps_arr,threshold,sample_beginning):
    fichier={}
    tab_FCF=[tab[i][0] for i in range(len(tab))]
    tab_FCPP=[tab[i][1] for i in range(len(tab))]
    seizures=[]
    background=[]
    en_background=True
    new_background=[sample_beginning+temps_arr[0]*60]      
    for i in range(len(tab_FCF)):
        if tab_FCF[i]/tab_FCPP[i]>=threshold and en_background: #detection 1er 1
            new_background.append(sample_beginning+temps_arr[i]*60)
            background.append(new_background)
            en_background=False
            new_seizure=[sample_beginning+temps_arr[i]*60]
        elif tab_FCF[i]/tab_FCPP[i]>=threshold : # en crise ( suite de 1)
            '''do nothing'''
    
        else:
            if not en_background: # sortie de crise
                new_seizure.append(sample_beginning+temps_arr[i]*60)
                seizures.append(new_seizure)
                en_background=True
                new_background=[sample_beginning+temps_arr[i]*60]
    new_background.append(sample_beginning+temps_arr[-1]*60)
    background.append(new_background)
    
    background_merged=[background[0]]                #On fusionne les détections de crises proches
    seizures_merged=[]
    #print(background_merged)
    if len(seizures)>0:
        seizures_merged.append(seizures[0])
        index_merged=[]
        for k in range(len(seizures[1:])):
            new_seizure=seizures[k+1]
            t_start=new_seizure[0]
            if t_start - seizures_merged[-1][1]<10:
                seizures_merged[-1][1]=new_seizure[1]
                #print(seizures_merged[-1])
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
    
    
    #fichier["background"]=background
    #fichier["seizure"]=seizures
    return(fichier)


#fisher(rr_path)
fisher('../database_processed_clean/00001981/rr_00001981_s001_t000.csv')
