import csv
import time
import pandas as pd
# Local Imports
import apps
import parser

def main(app: apps.App) -> None:
    start = time.perf_counter()
    print(f"\n➡️  Processing Reviews for {app.name} [{app.id}]")
    df = parser.load_app_reviews(app, 'english')
    
    count = len(df)
    # df.info()
    # print('\n',df['review'].head(10))

    # Prepare data
    df["date"] = pd.to_datetime(df["timestamp_updated"], unit="s")
    
    df['review'] = df['review'].apply(parser.preprocess_text)
    
    df_timestamp_zero = df[df["timestamp_created"] == 0]
    if len(df_timestamp_zero) > 0:
        print(f"⚠️  {len(df_timestamp_zero)} out of {count} reviews have 'timestamp_created == 0'")
        print(f"{df_timestamp_zero}")

    df.to_csv(f'./csv/{app.id}_processed.csv', quoting=csv.QUOTE_NONNUMERIC)
    
    duration = time.perf_counter()- start
    print(f"⏱️  Processed {count} Reviews in {duration:.2f} seconds [{count / duration:.2f} Reviews per sec]")
    

if __name__=='__main__':
    print(f"⏰ Starting at {time.strftime("%d/%m/%Y %H:%M %p",time.localtime())}")
    start = time.perf_counter()

    # main(apps.get_app_by_name('Dota 2'))
    # main(apps.get_app_by_id('413150'))
    [main(app) for app in apps.apps]
    
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start:.2f} seconds\n")