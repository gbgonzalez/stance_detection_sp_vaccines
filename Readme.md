# Model to detect stance in Spanish tweets about of COVID-19 vaccination  


This repository provides the training and testing of the BERTIN+BiLSTM model for vaccination stance detection in Spanish tweets related with COVID-19. Moreover, this repository contains a script to hydrate tweets, i.e. to use the Twitter API to extract the corpus.

## Installation

The following section describes the installation process. This program has been tested on Python version 3.9. To download the repository as well as to install the necessary libraries, it is necessary to execute:

```bash
$ git clone https://github.com/gbgonzalez/stance_detection_sp_vaccines.git
$ cd stance_detection_sp_vaccines; pip install -r requirements.txt
```

## Execution
Once in the root folder of the repository it may be executed using the following commands:
### Hydrated tweets
For hydrate corpus tweets:
```bash
$ python app\run.py -o hydrated_tweets
```
### Train model
First, we select the optimal parameters:
<ul>
    <li> <b> batch_size: </b> corresponds to the batch size with which the model will be trained </li>
    <li> <b> lr: </b>corresponds to the learning rate with which the model will be trained</li>
    <li> <b> epochs: </b> number of epochs with which the model will be trained</li>
    <li> <b> max_len: </b> maximum length of each of the documents in the corpus</li>
    <li> <b> d: </b>  correspond to the dropout value</li>
    <li> <b> ae: </b> value of Adam Epsilon</li>
</ul>

For example:
```bash
$ python app\run.py -o train -max_len 128 -batch_size 16 -lr 3e-5 -epochs 4 -d 0.3 -ae 1e-8 
```

### Test model
Then, we select the following parameters:
<ul>
    <li> <b> batch_size: </b> corresponds to the batch size with which the model will be trained </li> 
    <li> <b> max_len: </b> maximum length of each of the documents in the corpus</li>
</ul>

For example:
```bash
$ python app\run.py -o test -max_len 128 -batch_size 32
```


All the information about the arguments used in the program can be displayed using:
```bash
$ python app\run.py -h
```

## Results

The BERTIN+BiLSTM achieved the following results:

Model | AVG F1-Score | F1 - Against | F1 - None | F1 - In favour
------------- |----------|--------------|-----------| ------------- 
BERTIN+BiLSTM | 0.87 | 0.88         | 0.92      | 0.80

