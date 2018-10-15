CS 7641 Machine Learning
Fall 2018 - OMSCS
mmendoza32

How to run project codes:

Dataset 1: Credit Card Fraud Detection

  1. Download data from Kaggle (login needed)
     https://www.kaggle.com/mlg-ulb/creditcardfraud/home
     downloaded file: creditcardfraud.zip
  
  2. Unzip the downloaded file. The zip file contains a single csv file, creditcard.csv.

  3. Place the unzipped file, creditcard.csv, in the same directory as dataset1.py.
     (dataset1.py is in student main directory, mmendoza32)

  3. Run python script, dataset1.py
     $ python dataset1.py
  
  * dataset1.py takes 30 minutes to run and generates all the png files used in the report.
  ** there is no separate test dataset file. the code will split the original csv file and set aside a test set.

Dataset 2: Sign Language MNIST

  1. Download data from Kaggle (login needed)
     https://www.kaggle.com/datamunge/sign-language-mnist/home
     downloaded file: sign-language-mnist.zip
 
  2. Unzip the downloaded file, extracting the csv files only.
     $ unzip sign-language-mnist.zip sign_mnist_*.csv
     Archive:  sign-language-mnist.zip
       inflating: sign_mnist_test.csv     
       inflating: sign_mnist_train.csv    

  3. Place the files, sign_mnist_train.csv and sign_mnist_test.csv,  
     in the same directory as dataset2.py (student main directory, mmendoza32)

  4. Run python script, dataset2.py
     $ python dataset2.py

Notes:
  - All relevant files are in my main directory. There are no subdirectories.
  - Versions used:
     python = 2.7,15
     sklearn = 0.19.1
     pandas = 0.23.4
     numpy = 1.15.1
     matplotlib = 2.2.3
  
