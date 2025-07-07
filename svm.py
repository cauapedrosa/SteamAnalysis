import pandas as pd
import parser as p
import apps
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    df = pd.DataFrame()
    dfs = []
    for app in apps.apps:
        df_aux = p.load_preprocessed_app_reviews(app)
        dfs.append(df_aux)
        # print(f"Loaded {len(df)} reviews for app {app.name} [{app.id}]")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total Reviews loaded: {len(df)}")


    # Features
    X = df[["review", "author_playtime_forever", "author_num_games_owned", "steam_purchase", "received_for_free"]]
    y = df["voted_up"].astype(int)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(stop_words="english"), "review"),
            ("num", StandardScaler(), ["author_playtime_forever", "author_num_games_owned"]),
            ("bool", OneHotEncoder(handle_unknown="ignore"), ["steam_purchase", "received_for_free"])
        ]
    )

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC(max_iter=10000))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    print(classification_report(y_test, y_pred))

    # # See what went wrong
    # wrong = X_test[y_pred != y_test]
    # print(wrong)

if __name__=='__main__':
    # for app in apps.apps: main(app) 
    # app = apps.apps_by_name['PUBG: BATTLEGROUNDS']
    main()