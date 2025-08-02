import time
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Local imports
import parser as p
import apps

# Setup Caching for Performance in Tests
from joblib import Memory
cachedir = './sklearn_cache'
memory = Memory(location=cachedir, verbose=1)

def load_data() -> pd.DataFrame:
    df = pd.DataFrame()
    dfs:list[pd.DataFrame] = []
    
    # ALL APPS
    for app in apps.apps:
        df_aux = p.load_preprocessed_app_reviews(app)
        dfs.append(df_aux)
        print(f"Loaded {len(df_aux)} reviews for app {app.name} [{app.id}]")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total Reviews loaded: {len(df)}\n")

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

    # Drop NA reviews
    print(f"> Dropping {df['review'].isna().sum()} missing values...")
    df = df.dropna(subset=['review'])
    print(f"> After dropna df has {len(df)} reviews...") # DEBUG
    return df

def train_svm_model(X_train, y_train) -> Pipeline:
    # Preprocessing
    print("\n> Setting up Model...")
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
            ("classifier", LinearSVC(max_iter=10000))
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
    print("\n> Setting up Model...")
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
                n_estimators = 100,     # Default = 100
                max_depth = 50,         # Default = None
                random_state = 0,       # Default = None
                n_jobs = 12,            # Default = None (Use only 1 Core) | -1 Uses all Cores
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
        random_state=42
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
    # models.append(rf_model)
    
    # for model in models: #TODO: Implement Model-testing loop
    
    # Predict
    print("\n> Predicting...")
    # y_pred = svm_model.predict(X_test)    # SVM
    y_pred = rf_model.predict(X_test)       # RF
    
    # Evaluate
    print("\n> Evaluating Predictions...")
    class_report = classification_report(y_test, y_pred)
    # output_classification_report(class_report, './results/svm.txt', f'{svm_model}')
    output_classification_report(class_report, './results/rf.txt', f'{rf_model}')
    
    # See what went wrong
    print("\n> Seeing what went wrong...")
    wrong = X_test[y_pred != y_test]
    print(wrong)

if __name__=='__main__':
    print(f"⏰ Starting at {time.strftime("%d/%m/%Y %H:%M %p",time.localtime())}")
    start = time.perf_counter()    
    # for app in apps.apps: main(app) 
    # app = apps.apps_by_name['PUBG: BATTLEGROUNDS']
    main()
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start :.2f} seconds\n")
