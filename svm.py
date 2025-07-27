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

# Setup Caching for Performance in Tests
from joblib import Memory
cachedir = './sklearn_cache'
memory = Memory(location=cachedir, verbose=1)


def main():
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
    print("> Dropping unused columns...")
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

    # Features
    print("\n> Starting Model Definition...")
    X = df[["review", "author_playtime_forever", "author_num_games_owned", "steam_purchase"]]
    y = df["voted_up"].astype(int)

    # Preprocessing
    print("> Preprocessing...")
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
    print("> Setting up Pipeline...")
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC(max_iter=10000))
        ],
        verbose=True,
        memory=memory,
    )

    # Split
    print("> Splitting Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    print(f"Train X:{len(X_train)} | Y:{len(y_train)}")
    print(f"Test X:{len(X_test)} | Y:{len(y_test)}")

    # Train
    print("> Fitting Model...")
    pipeline.fit(X_train, y_train)
    
    # Pickle model
    print("> Pickling Model...")
    with open('./models/svm_2.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    # Predict
    print("> Predicting...")
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    print("> Evaluating Predictions...")
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    with open('./results/svm_1.txt', 'w+') as f:
        f.write(str(class_report))

    # See what went wrong
    print("> Seeing what went wrong...")
    wrong = X_test[y_pred != y_test]
    print(wrong)

if __name__=='__main__':
    print(f"⏰ Starting at {time.strftime("%d/%m/%Y %H:%M %p",time.localtime())}")
    start = time.perf_counter()    
    # for app in apps.apps: main(app) 
    # app = apps.apps_by_name['PUBG: BATTLEGROUNDS']
    main()
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start :.2f} seconds\n")
