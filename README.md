# TODO
[] re-construct with package malwareDetector
[] modify the input format
	1. batch
	2. type, include bin, raw feature, feature vector
[] testing

# Graph Theory Detector

> ref. ALASMARY, Hisham, et al. Analyzing and detecting emerging Internet of Things malware: A graph-based approach. *IEEE Internet of Things Journal*, 2019, 6.5: 8977-8988.

## Introduction

### Description

* The main program is a malware detector.

  * **Input** : a binary file
  * **Output** : prob. of each class predicted by ML model
  * **Flow** : 
    ![](flow.jpg)
    * reverse the bin and extract the feature
    * load the model
    * predict
    * print the result
    * write the result -> output csv file

### Feature Extraction

* We reverse the binary file to function call graph(FCG) by r2pipe, then extract the attribute of FCG:

  * number of nodes

  * number of edges

  * density

  * (mean, max, min, median, std) of closeness_centrality


  * (mean, max, min, median, std) of betweenness_centrality


  * (mean, max, min, median, std) of degree_centrality


  * (mean, max, min, median, std) of shortestpaths.avglen 

  The dimension of feature is 23.


## Requirements

* python3
* radare2
* python package
  * r2pipe
  * networkx
  * joblib
  * argparse
  * sklearn

## Files

* **FeatureConstruction** : some code about gpickle to feature.csv (for training, validation)
* **Modeling** : some code about training and saving the model
* **MD(FC)_Model** : save the model with .joblib, e.g. `rf.joblib`
* **TestingBin** : some binary for testing
* **main.py** : the detector(classifier)
* **graphity.py** : module for feature extraction
* **utils.py** : some useful func.
* **scaler.joblib** : scale the feature vector

## Usage
* unzip models to `FC_Model` and `MD_Model`: `make`
* input binary: `-i <path>`, `--input-path <path>`
* model: `-m <model>`, `--model <model>`
  * rf, mlp, knn, svm
* output (record): `-o <path>`, `--output-path <path>`
* Malware Detection / Family Classification
    * do nothing if you wanna do malware detection(binary clf)  
    * add `-c` if you wanna do family classification 
* e.g.
    `python main.py -i testingBin/0021eaf2 -o myDetector_FC_records.csv -m rf -c`
    * using trained rf family classifier(`-c`), predict '0021eaf2' and write the result to 'myDetector_FC_records.csv'
    * add `-W ignore` if you keep getting bothered by warning msg
### note
* a output csv file for a experiment
  * e.g.
    
    ```python=
    for bin in bins:
      cmd = 'python main.py -i ' + bin
      cmd += ' -m rf -o GraphTheoryDetector_RF_MD.csv'
      os.system(cmd)
    ```
    specify another output file if you wanna do FC task or you wanna change the model
* output file format

  |    Filename  | Benignware | Malware |
  | :----------: | :------: | :-------: |
  | 00ffe391     |   0.97   |   0.03    |
  |     00f391fe      |  0.967   |  165.51   |
  |     1fe00f39      |  -1   |    |
  * it will record the prob of each class
  * -1 means fail

## Model Performance
* Malware Detection
  |    Models    | Accuracy | Time cost |
  | :----------: | :------: | :-------: |
  | RandomForest |   0.97   |   5.01    |
  |     KNN      |  0.967   |  165.51   |
  |     SVM      |  0.772   |  1606.21  |
  |     MLP      |  0.879   |   49.51   |

* Family Classification
  |    Models    | Accuracy | Time cost |
  | :----------: | :------: | :-------: |
  | RandomForest |  0.952   |   4.21    |
  |     KNN      |  0.944   |  104.92   |
  |     SVM      |   0.71   |  1562.06  |
  |     MLP      |  0.887   |   70.99   |
