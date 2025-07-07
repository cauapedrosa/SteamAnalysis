import apps
import parser as p
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = f"plot2.png"
FILE_PATH = f"./figures/{FILE_NAME}"
def main():
    start = time.perf_counter()    
    
    
    # Load JSON data
    df = p.load_all_app_raw_reviews()

    # Prepare data
    df = df[df["appname"].notnull()]
    df["appname"] = df["appname"].astype(str)
    df["date"] = pd.to_datetime(df["timestamp_updated"], unit="s")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    
    # # Scatter Plot
    grouped = df.groupby(["appname", "month"]).size().reset_index(name="review_count")
    np.random.seed(42)
    jitter = np.random.uniform(-2, 2, size=len(grouped))  # small random shift

    grouped["review_count_jitter"] = grouped["review_count"] + jitter
    cat = grouped["appname"].astype("category")
    grouped["color_id"] = cat.cat.codes  # numeric color values

    plt.figure(figsize=(14, 6))
    scatter = plt.scatter(
        grouped["month"],
        grouped["review_count_jitter"],
        c=grouped["color_id"],
        cmap="tab10",
        # cmap="hsv",
        alpha=0.7,
        s=10,
    )

    plt.title("Monthly Review Counts per Game (Scatter Plot with Jitter)")
    # plt.xlabel("Time")
    plt.ylabel("Number of Reviews (jittered)")
    plt.grid(True)
    plt.tight_layout()

    handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
    labels = cat.cat.categories.tolist()
    max_labels = 10
    # plt.legend(handles[:max_labels], labels[:max_labels], title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.legend(handles, labels, title="Game", loc="upper left")
    plt.show(block=False)
    plt.savefig(FILE_PATH) 
    print(f"\n⏱️  Total execution time: {time.perf_counter() - start :.2f} seconds\n")


if __name__=='__main__':
    main()