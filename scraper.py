import os
import traceback
import pandas as pd
import requests
import json
import time
from apps import *

DEBUG = False

def main():
    try:
        langs_list= [
            'english',
            # 'brazilian',
            ]
        
        # [fetch_reviews(app) for app in apps if app.id] # All Apps
        
        # [fetch_reviews(app) for app in apps if app.id not in ['730','252490', '578080', '1063730']] # All Apps except in list
        
        # fetch_reviews(get_app_by_name("Dota 2"), verbose=DEBUG) # Single App by Name
        
        fetch_reviews(get_app_by_id("413150"), verbose=DEBUG) # Single App by ID
    except Exception as e:
        print(f"⚠️ Scraper main() Raised {e}!")
        print(traceback.print_exc())        
    return

def fetch_reviews(app: App, language='english', max_pages=99999, verbose=False):
    print(f"Fetching Reviews for {app.name} [{app.id}]\nLanguage: {language} | max_pages: {max_pages} | verbose: {verbose}")

    # Declare Variables
    url = f"http://store.steampowered.com/appreviews/{app.id}/"
    params = {
    'json':'1', 
    'num_per_page':'100',
    'filter': 'all',
    # 'filter': 'recent',
    # 'day_range':'365',
    'filter_offtopic_activity': 0,
    'language': language,
    'cursor': "*"
    }

    all_reviews = []
    seen_cursors = set()

    # Start Data Pulling
    for page in range(max_pages):
        if verbose: print(f"\nFetching Page #{page} | Cursor: {params['cursor']}")
        try:
            resp = requests.request("GET", url=url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"⚠️ fetch_reviews Raised {e}!")
            print(traceback.print_exc())
            break

        if verbose: print(f"> Requesting: {resp.request.url!r}") # DEBUG
    
        # Stop if no reviews received
        page_reviews = data.get('reviews', [])
        if not page_reviews:
            print(f"⚠️\tPage #{page} contains no reviews; stopping.")
            break
        else:
            all_reviews.extend(page_reviews)

        raw_cursor = data['cursor']
        if raw_cursor in seen_cursors:
            print("🔄\tDetected repeat cursor; stopping.")
            break
        
        seen_cursors.add(raw_cursor)
        params['cursor'] = raw_cursor

        if verbose: print(f"Page #{page} has page_reviews: {len(page_reviews)}") # DEBUG
        print(f"{app.name} [{app.id}] | Page #{page} | Found {len(all_reviews)} reviews so far...")
        # sleep(1)

    add_to_json(all_reviews, app, language)

def add_to_json(list_reviews_to_add: list, app: App, language='english'):
    df_reviews_to_add = pd.DataFrame(list_reviews_to_add)
    if df_reviews_to_add.empty: # Guard Clause for Empty Reviews Param
        print(f"⚠️ Skipped app {app.id} — no reviews to add.")
        return
    
    file_path = f"./json/{app.id}-{language}.json"
    
    try:
        if os.path.exists(file_path):
            df_reviews_loaded = pd.read_json(file_path, orient='records')
        else: df_reviews_loaded = pd.DataFrame()

        if DEBUG: print(f"# Found {len(df_reviews_loaded)} reviews in {file_path}")
        if DEBUG: print(f"# Concatenating {len(list_reviews_to_add)} reviews to file")

        df_reviews_combined = pd.concat([df_reviews_loaded, df_reviews_to_add])

        if DEBUG: print(f"# Concat resulted in {len(df_reviews_combined)} reviews in DF")

        df_reviews_combined['recommendationid'] = df_reviews_combined['recommendationid'].astype(str)
        
        df_reviews_combined["timestamp_created"] = pd.to_numeric(
            df_reviews_combined["timestamp_created"], errors="coerce"
        ).fillna(0).astype(int)
        
        df_reviews_combined["timestamp_updated"] = pd.to_numeric(
            df_reviews_combined["timestamp_updated"], errors="coerce"
        ).fillna(0).astype(int)        
        
        df_reviews_combined.drop_duplicates(
            subset='recommendationid',
            keep='last',
            inplace=True)
        if DEBUG: print(f"# After drop_duplicates(): {len(df_reviews_combined)} reviews in DF")
          
        df_reviews_combined.to_json(file_path, orient="records", indent=4, force_ascii=False)
        
        print(f"\n💾\tSaved {len(df_reviews_combined)} reviews into {file_path}")

    except Exception as e:
        print(f"⚠️ add_to_json({app.id},{language}) raised {e}!")
        raise e
    
    return


if __name__=='__main__':
    print(f"⏰ Starting at {time.strftime("%d/%m/%Y %H:%M %p",time.localtime())}")
    start = time.perf_counter()    
    main()
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start :.2f} seconds\n")
