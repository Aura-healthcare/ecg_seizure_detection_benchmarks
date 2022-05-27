# ecg_seizure_detection_benchmarks

Ce github est celui du projet P1.01 du pôle IA de CentraleSupélec, réalisé avec l'association Aura. Sa structure se base sur celle du Github principal d'Aura.
L'objectif de ce projet a été d'implémenter des algorithmes de détection des crises d'épilepsie décrits comme performants dans la littérature, puis d'évaluer leurs performances afin de pouvoir comparer leur efficacité.

3 algorithmes ont été implémentés :

* Fisher -- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5064739/#ner12376-bib-0017
* Jeppesen -- Seizure detection based on heart rate variability using a wearable electrocardiography device
* Vandercasteele -- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5676949/

Tous les scripts Python codés dans le cadre de ce projet son situés dans le dossier src > projet_CS

## Getting started

Type the command <git clone https://github.com/louislhotte/Projet-Epilepsie.git > into your interface 

### Dependencies

The pipeline requires these packages to run :
* airflow-provider-great-expectations==0.0.7
* biosppy==0.7.3
* click==8.0.1
* ecg-qc==1.0b5
* great-expectations==0.13.25
* hrv-analysis==1.0.4
* joblib==1.1.0
* json == 1.0.2
* mlflow==1.21.0
* matplotlib==3.5.1
* numpy==1.19.5
* pandas==1.1.5
* psycopg2-binary==2.8.6
* py-ecg-detectors==1.0.2
* pyEDFlib==0.1.22
* scikit-learn==1.0.2
* scipy==1.7.2
* seaborn==0.11.2
* statistics == 0.0.7
* wfdb==3.4.0

You can install them in a virtual environment on your machine via the command : 
```sh
    $ pip install -r requirements.txt
```

## Explanation of how the differents algorithms work

### Fisher
How does Fisher's algorithm work ?
Fisher is based on the values of RR interval lengths, that allow to calculate mean heart rate. 
Two different means are calculated : 
* a "background" HR mean, calculated by averaging the HR of the patient on a "large window", of length 5 min by default.
* a "leading" HR mean, calculated by averaging the HR of the patient on a "short window", of length 10 sec by default.
We calculate then (leading HR mean)/(bckg HR mean) and if the value is superior to the threshold, we consider that a seizure is detected. This very simple algorithm has already been used in the VNS Therapy System.

### Jeppesen
How does Jeppesen's algorithm work ?
Jeppesen is based on features, which are calculated from the values of RR interval lengths. For example, the mean value of all RR interval lengths would be a feature (a bad feature, but still a feature). 
Let's stay we choose a sliding window of a 100 RR intervals : that means, for each new RR interval, we will calculate the value of the feature we chose on the last 100 RR intervals. That will give us a time-dependent value of this feature.
Now, we have to define thresholds for our features. Here, they will be recalculated for each new recording. In reality, we only need to calculate them once for each patient.
This threshold will be 1.05*(Max value of the feature on a period without seizures). For pragmatic and scientific reasons, on this dataset, we chose to calculate this threshold on the following period : from the beginning of the recording until 1 minute before the first seizure.
A seizure is detected if the value of the feature overlaps the threshold. To avoid detecting a single seizure more than once, the system can't detect a new seizure for the next 3 minutes.

### Vandercasteele


## Specific scripts of this project

### Format of the data in input & output 
The dataset that you'll use needs to be preprocessed : as the algorithms only work on samples under the format of .csv files with rr_intervals (columns format :  ) with their annotation in a .json file, the original .edf samples and annotations have to go through a preprocess pipeline. You can for instance use Aura's pipeline with the script 1_detect_qrs_wrapper.sh.

Example of a rr .csv file :
<img src="https://github.com/Aura-healthcare/ecg_seizure_detection_benchmarks/tree/main/Images/screenshot_rrintervals.png"/>

Example of an annotation .json file :
<img src="https://github.com/Aura-healthcare/ecg_seizure_detection_benchmarks/tree/main/Images/screenshot_json.png"/>

### Executing Fisher on a rr_file 

/!\ The sample need to have a length longer as the LARGE_WINDOW you choose !!

You can directly execute Fisher algorithm on a rr_file. The result will be the form of a output.json file (with the same format as the annotations) that you can find in the detection_algorithms folder. You can also change several parameters of the Fisher algorithm, such as the length of the long and short windows (in ms), the value of threshold (1.4 = 40% of threshold), the begininng and the end of the part of the sample on which you want to execute Fisher.

For example :

```sh
    $ python3 src/projet_CS/detection_algorithms/fisher.py --rr-path <path-to-the-rr-file> --threshold <threshold> --long-window <length-of-the-long-window-in-ms>
```

### Executing Jeppesen on a rr_file 

How to use Jeppesen's algorithm :
As Jeppesen is based on features, first step is to choose a feature, for example 'csi'
Second step is to enter the RR recording you want to analyze, but also the associated annotation under json format. In fact, we need it to calculate the threshold.
You can also choose to change the size of the sliding-window, or not.

For example :

```sh
python3 -m detection_algorithms.Jeppesen.Jeppesen --rr-path 'rr_pathname' --annotation-path 'jsonannotation_pathname' --jeppesen-feature-name 'csi' --sliding-window 100
```
The result will be the form of a output.json file (with the same format as the annotations) that you can find in the detection_algorithms folder.

The algorithm is coded so that you can easily add your own features, by adding them in the "custom_features" file, located in the 'Jeppesen' folder.

### Executing Vandercasteele on a rr_file 

Vandercasteele is a ML algorithm. It therefore has to be trained before one can use it.

Training the model : 
    You have to have an annotated seizure database to train the model, of the form similar to the one in this repository, under database_processed_clean.
    The parameters of the models can be tweeked directly in the code under the function create_svm(). You can change the values of Gamma and C there.
    Default values are used if not.
   to train the model, type in the bash console, when under seizure_detection_pipeline/ : 
   
```sh
python3 src/Projet_CS/detection_algorithms/vandercasteele.py --db_train_path 'db_train_path'
```
   
   where 'db_train_path' is the path of the database you want the algorithm to train on.

How to execute the algorithm : 
    You can directly execute vandercasteele algorithm on a rr_file. The result will be the form of a output.json file, where you can find a dictionary containing       all the start time of the seizures it detected. If the dictionary is empty, no seizures have been detected. The output.json can be found in the                     detection_algorithms folder. It will use the model trained that is on path models/model_centralesupelec.joblib ( still under seizure_detection_pipeline/).

For example ; 
```sh
python3 src/Projet_CS/detection_algorithms/vandercasteele.py --csv_path 'csv_path'
```
 where 'csv_path' is the path of the csv you want to pass through.
 
 WARNING : do not train and evaluate vandercasteele at the same time. 
 
### Evaluate a dataset

You can directly evaluate one algorithm on a dataset made up of folders for each patient with rr_files (rr_patient_number_sample_id.json) and annotations  (patient_number_sample_id.json). If you want to evaluate Fisher, the samples need to be longer than the LARGE_WINDOW you choose.

```sh
    $ python3 src/projet_CS/evaluate.py --db-path <path-to-the-database-to-evaluate> --algorithm <algorithm-to-evaluate-'fisher'-'jeppesen'-'vandercasteele'> --jeppesen-feature-name <name-of-the-jeppesen-feature-to-evaluate>
```

The results of the evaluation will be displayed on your screen, and then saved in the folder "models_evaluation".


## The team

* Thibault Novat
* Antoine Bohin
* Louis Lhotte
* Maxime Vanderbeken
* Paul Lavandier
* William Slimi
