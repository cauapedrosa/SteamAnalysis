import time
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Local imports
import parser as p
import apps

def main():
    # SVM 1
    with open('models/svm_1.pkl', 'rb') as file:
        svm_1:Pipeline = pickle.load(file)
    print(svm_1)


if __name__=='__main__':
    print(f"⏰ Starting at {time.strftime("%d/%m/%Y %H:%M %p",time.localtime())}")
    start = time.perf_counter()    
    # for app in apps.apps: main(app) 
    # app = apps.apps_by_name['PUBG: BATTLEGROUNDS']
    main()
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start :.2f} seconds\n")
