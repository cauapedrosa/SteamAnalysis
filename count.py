import os
import pandas as pd
import time
import parser as p
import apps

_total_review_count:int = 0
_total_positive_review_count:int = 0
_total_negative_review_count:int = 0

def count(app) -> int:
    '''Gets the review count of an App
    
    Args:
        App (app): The App to count Reviews for
        Processed (bool): To load Processed reviews from CSv or Raw reviews from JSON
    
    Returns:
        int: The total number of review records found for the App
    '''
    global _total_positive_review_count, _total_negative_review_count
    app_review_count: int = 0
    try:
        print(f"\n\t➡️  {app.name} [ID: {app.id}]: ")
        
        # Load Reviews into 'df'
        file_path:str = f"./json/{app.id}-english.json"
        df = p.load_app_reviews(app, 'english')
        
        # Count Rows of Reviews        
        app_review_count:int = len(df)
        
        # Abort if 0 reviews found
        if app_review_count == 0:
            print(f"{file_path} exists ? {os.path.exists(file_path)}")
            raise Exception(f"⚠️  No reviews for {app.name} were found. Skipping...")

        # Count Positive / Negative Reviews
        val_counts = df['voted_up'].value_counts()
        val_positive = val_counts.get(True, 0)
        val_negative = val_counts.get(False, 0)

        # Add Current App's reviews to script-wide counters
        _total_positive_review_count += val_positive
        _total_negative_review_count += val_negative

        # Display Counts
        ratio = val_positive / val_negative if val_negative != 0 else float('inf')
        print(f"App Reviews: {app_review_count:6d} | Positive: {val_positive:6d} | Negative: {val_negative:6d} | Ratio: {ratio:2f}")
        if val_positive + val_negative != df['voted_up'].notna().sum():
            raise Exception("Values for Positive and Negative Reviews don't add up to Total Reviews")

    except FileNotFoundError as e:
        print(f"⚠️  FileNotFound! No reviews for {app.name} were found.\n{e}")
    except Exception as e:
        print(e)

    return app_review_count

def main() -> None:
    global _total_positive_review_count, _total_negative_review_count, _total_review_count

    app_list_to_count = apps.apps
    # app_list_to_count = [apps.get_app_by_id('570')]
    
    start = time.perf_counter()
    
    print(f"\nCounting reviews for {len(app_list_to_count)} apps:")
    for app in app_list_to_count:
        # count(app)
        _total_review_count += count(app)
    print("\n" + "█" + "▀"*50 + "█")
    print(f"█ ✅  Total Review Count: {_total_review_count:^24d} █")
    print("█" + " "*50 + "█")
    print(f"█ Positive: {_total_positive_review_count:^15d} Negative: {_total_negative_review_count:^12d} █")
    print("█" + " "*50 + "█")
    print(f"█ General Ratio: {(_total_positive_review_count / _total_negative_review_count):>26f}:1      █")
    print("█" + " "*50 + "█")
    print(f"█ Average Review Count per App ({len(app_list_to_count)}): {_total_review_count / len(app_list_to_count) :^12.1f}  █")
    print("█" + "▄"*50 + "█")
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start:.2f} seconds\n")

if __name__=='__main__':
    main()