import parser
import apps
import time
import pandas as pd


def main(app: apps.App = apps.get_app_by_id('570')) -> None:
    start = time.perf_counter()
    
    # print(f"\n> Loading reviews all {len(apps.apps)} apps")
    # df = parser.load_all_app_raw_reviews()
    
    print(f"\n> Loading reviews for {app.name} [{app.id}]")
    df = parser.load_app_reviews(app, 'english')
    # df = parser.load_preprocessed_app_reviews(app)    
    

    # Prepare data
    df["appname"] = app.name
    df["date"] = pd.to_datetime(df["timestamp_updated"], unit="s")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    
    # df['review'] = df['review'].apply(parser.preprocess_text)
    
    print(f"{df[df["timestamp_created"] == 0]}")
    
    print(f"\n{df['timestamp_created'].dt.to_period('M').value_counts().sort_index()}")
    print(f"\n{df['timestamp_updated'].dt.to_period('M').value_counts().sort_index()}")
    # print(f"\n{df['date'].dt.to_period('M').value_counts().sort_index()}")
    # print(f"\n{df['month'].dt.to_period('M').value_counts().sort_index()}")


    print(f"Min/Max:\n{df[['timestamp_created', 'timestamp_updated']].agg(['min', 'max'])}")

    
    df.to_csv(f'./test.csv')
    # df.to_csv(f'./csv/{app.id}_processed.csv')
    
    print(f"\n⏱️  Execution time for {app.name}: {time.perf_counter()- start:.2f} seconds")

if __name__=='__main__':
    print(f"⏰ Starting at {time.strftime("%d/%m/%Y %H:%M %p",time.localtime())}")
    start = time.perf_counter()

    main()
    # [main(app) for app in apps.apps]

    print(f"\n⏱️  Total execution time: {time.perf_counter() - start:.2f} seconds\n")