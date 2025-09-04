import time
import pickle
import pandas as pd
import traceback
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Local imports
import parser as p
import apps

# Setup Caching for Performance in Tests
from joblib import Memory
cachedir = './sklearn_cache'
memory = Memory(location=cachedir, verbose=1)
start = time.perf_counter()

def load_data() -> pd.DataFrame:
    df = pd.DataFrame()
    dfs:list[pd.DataFrame] = []
    
    # ALL APPS
    for app in apps.apps:
        df_aux = p.load_preprocessed_app_reviews(app)
        dfs.append(df_aux)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n> Loaded: {len(df)} reviews in Total!\n")

    # SINGLE APP
    # app = apps.get_app_by_id('570')
    # df = p.load_preprocessed_app_reviews(app)
    # print(f"Loaded {len(df)} reviews for app {app.name} [{app.id}]")
    
    # Drop Unused Columns
    print("\n> Dropping unused columns...")
    unused_columns = [
        "votes_up",
        "votes_funny",
        "weighted_vote_score",
        "comment_count",
        "received_for_free",
        "written_during_early_access",
        "primarily_steam_deck",
        "timestamp_created",
        "timestamp_updated",
        "author_playtime_last_two_weeks",
        "author_deck_playtime_at_review",
        "author_last_played",
    ]
    df = df.drop(columns=unused_columns)
    print(f"> After dropping unused columns df has {len(df)} reviews...") # DEBUG

    # Drop NA reviews
    print(f"> Dropping {df['review'].isna().sum()} missing values...")
    df = df.dropna(subset=['review'])
    print(f"> After dropna df has {len(df)} reviews...") # DEBUG
    
    print(f"⏱️  Total loading time: {time.perf_counter() - start :.2f} seconds\n")    
    return df

def train_svm_model(X_train, y_train) -> Pipeline:
    # Preprocessing
    print("\n> Starting Training of LinearSVC Model...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                stop_words="english",
                ngram_range=(2,3)), # Bigrams & Trigrams
                "review",
            ),
            ("num", StandardScaler(), ["author_playtime_forever", "author_num_games_owned"]),
            ("bool", OneHotEncoder(handle_unknown="ignore"), ["steam_purchase"])
        ]
    )
    print(f"> Preprocessor: {preprocessor}")

    # Pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC(
                max_iter=10000
            ))
        ],
        verbose=True,
        memory=memory,
    )
    print(f"> Pipeline: {pipeline}")

    # Fit model on Train data
    print("\n> Fitting Model...")
    pipeline.fit(X_train, y_train)
    
    save_model(pipeline, './models/svm.pkl')
    return pipeline

def train_rf_model(X_train, y_train) -> Pipeline:
    # Preprocessing
    print("\n> Starting Training of Random Forest Classifier Model...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                stop_words="english",
                ngram_range=(2,3)), # Bigrams & Trigrams
                "review",
            ),
            ("num", StandardScaler(), ["author_playtime_forever", "author_num_games_owned"]),
            ("bool", OneHotEncoder(handle_unknown="ignore"), ["steam_purchase"])
        ]
    )

    # Pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators = 100,         # Default = 100
                max_depth = 100,            # Default = None
                class_weight = "balanced",  # Default = None
                random_state = 0,           # Default = None
                n_jobs = 12,                # Default = None (Use only 1 Core) | -1 Uses all Cores
                verbose = 2,
            ))
        ],
        verbose=True,
        memory=memory,
    )
    print(f"> Pipeline: {pipeline}")

    # Fit model on Train data
    print("\n> Fitting Model...")
    pipeline.fit(X_train, y_train)
    
    save_model(pipeline, './models/rf.pkl')
    
    return pipeline

def load_model(file_name:str) -> Pipeline:
    print(f"\n> Loading Model from {file_name}")
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(pipeline:Pipeline, file_name:str):
    print("\n> Pickling Model...")
    with open(file_name, 'wb') as f:
        pickle.dump(pipeline, f)
        print(f"Saved Pickled Model at {file_name}")
    return

def output_classification_report(classification_report, file_name, model_info='',):
    output =  ("\n" + "█" + "▀"*55 + "█")
    output += ("\n" + classification_report)
    output += ("█" + "▄"*55 + "█")
    print(output)
    
    with open(file_name, 'w+') as f:
        f.write(f"Model Info:\n{model_info}")
        f.write(f"\n" + "-"*50 + "\n")
        f.write(f"Classification Report:\n{str(classification_report)}")
        print(f"Saved Classification Report at {file_name}")

def main():
    df = load_data()
    
    # Features
    print("\n> Defining Features...")
    X = df[["review", "author_playtime_forever", "author_num_games_owned", "steam_purchase"]]
    y = df["voted_up"].astype(int)

    # Split
    print("\n> Splitting Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        # random_state=42
    )
    print(f"Train X:{len(X_train)} | Y:{len(y_train)}")
    print(f"Test X:{len(X_test)} | Y:{len(y_test)}")

    models:list[Pipeline] = []

    # > Train or Load SVM Model
    # svm_model = train_svm_model(X_train,y_train)
    # svm_model = load_model('./models/svm.pkl')    
    # models.append(svm_model)
    
    # > Train or Load RF Model
    rf_model = train_rf_model(X_train,y_train)
    # rf_model = load_model('./models/rf.pkl')
    models.append(rf_model)
    
    print(f"\n> Testing {len(models)} models...")
    for model in models:
        model_name = model.named_steps['classifier']
        print(f"> Testing {model_name}...")
        if isinstance(model_name , RandomForestClassifier): file_name = './results/rfc.txt'
        elif isinstance(model_name , LinearSVC): file_name = './results/svc.txt'
        else: raise Exception(f"Error: Unknown Model: {model}\n{type(model)}")        
        # Predict
        print("\n> Predicting...")
        y_pred = model.predict(X_test)
        
        # Evaluate
        print("\n> Evaluating Predictions...")
        class_report = classification_report(y_test, y_pred, zero_division=0)
        output_classification_report(class_report, file_name, f'{model}')
        
        print("> Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # See what went wrong
        # print("\n> Seeing what went wrong...")
        # wrong = X_test[y_pred != y_test]
        # print(wrong)

if __name__=='__main__':
    print(f"⏰ Starting at {time.strftime("%d/%m/%Y %H:%M %p",time.localtime())}")
    
    try:
        main()    
    except Exception as e:
        print("Execution Failed due to Exception: ", e)
        traceback.print_exc()
    
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start :.2f} seconds\n")
