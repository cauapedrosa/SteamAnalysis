import os
import apps
import json, csv
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
from wordcloud import WordCloud

# Import Langdetect + Set Seed #
from langdetect import detect, LangDetectException, DetectorFactory
DetectorFactory.seed = 0 # Make langdetect deterministic

# Set DEBUG #
DEBUG = False

# NLTK Download #
def download_if_missing(resource_name, resource_path):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=False)

download_if_missing('punkt', 'tokenizers/punkt')
download_if_missing('stopwords', 'corpora/stopwords')
download_if_missing('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
download_if_missing('wordnet', 'corpora/wordnet/lexnames')

# Start of Functions #

def load_all_app_raw_reviews(language:str = 'english') -> pd.DataFrame:
    """Loads All App Reviews from local JSONs
    
    Returns:
        pd.DataFrame: DataFrame containing all Review objects
    """

    df_output = pd.DataFrame()
    start = time.perf_counter()
    
    for app in apps.apps:
        # Load Data from JSONs into DFs
        file_path = f"./json/{app.id}-{language}.json"

        if os.path.exists(file_path):
            df = pd.read_json(file_path)
        else: df = pd.DataFrame()

        # If there are no reviews in the DataFrame created from read_json,
        if len(df) == 0:
            print(f"⚠️  {file_path} contains 0 reviews.") # warn user
            continue
        
        # Tag each review with its AppID
        df['appid'] = app.id
        df['appname'] = app.name
        
        df_output = pd.concat([df_output, df]).drop_duplicates(subset='recommendationid', keep='last')

    print(f"⏱️  Loaded {len(df_output)} reviews in  {time.perf_counter()-start:.2f} seconds")
    return df_output

def load_app_reviews(app: apps.App, language: str = 'english') -> pd.DataFrame:
    """Loads App Reviews from local json
    
    Args:
        App (app): The App to load Reviews for
        Language (str): The language to load reviews in. Valid values are "english" and "brazilian"
    
    Returns:
        pd.DataFrame: DataFrame containing all Review objects
    """
    start = time.perf_counter()
    
    # Load Data from JSONs into DFs
    file_path = f"./json/{app.id}-{language}.json"
    if os.path.exists(file_path):
        df = pd.read_json(file_path)
    else: raise Exception(f"⚠️  Could not load reviews for {app.name} because {file_path} doesn't exist. Skipping...") 

    # If there are no reviews in the DataFrame created from read_json,
    if len(df) == 0:
        print(f"⚠️  {file_path} contains 0 reviews.") # warn user
        return df # return empty df anyways
    else: # add AppID and Language Columns
        df['appid'] = app.id
        df['appname'] = app.name

    # Split JSON in 'Author' field into individual fields with 'author_' prefix.\
    author_df = pd.json_normalize(df['author']) # type: ignore
    author_df = author_df.add_prefix('author_')
    df = df.drop(columns=['author']).join(author_df)

    # Drop Unused Columns
    # df = df.drop(columns=[
    #     "votes_up",
    #     "votes_funny",
    #     "weighted_vote_score",
    #     "comment_count",
    #     "written_during_early_access",
    #     "primarily_steam_deck"
    # ])

    # Count DF Lines
    if DEBUG: print(f"# load_app_reviews() Returning {type(df)} with {len(df)} rows")

    print(f"> Loaded {len(df)} reviews in: ⏱️  {time.perf_counter()-start:.2f} seconds")
    return df

def load_preprocessed_app_reviews(app: apps.App, suffix: str = 'processed') -> pd.DataFrame:
    """
    Loads App Reviews from Preprocessed CSV
        
    Args:
        App (app): The App to load Reviews for
        Suffix (str): Used for composing Filename to Read: "./csv/{app.id}_{suffix}.csv". Default value is "processed".        
    
    Returns:
        pd.DataFrame: DataFrame containing all Review objects
    """
    start = time.perf_counter()
    
    # Load Data from JSONs into DFs
    path_reviews_json = f"./csv/{app.id}_{suffix}.csv"
    
    df = pd.read_csv(path_reviews_json, low_memory=False)  

    # If there are no reviews in the DataFrame created from read_json,
    if len(df) == 0:
        print(f"⚠️  {path_reviews_json} contains 0 reviews.") # warn user
        return df # return empty df anyways

    # Count DF Lines
    if DEBUG: print(f"# load_preprocessed_app_reviews() Returning {type(df)} with {len(df)} rows")
    
    print(f"> Loaded {len(df)} reviews in: ⏱️  {time.perf_counter()-start:.2f} seconds")
    return df

def get_wordnet_pos(tag):
    """
    Map NLTK POS tags to WordNet's format.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

def preprocess_text(text, language='english') -> str:
    """
    1. Lowercase
    2. Remove any character that isn’t a letter, digit or whitespace
    3. Tokenize
    4. Remove stopwords & any non-alphabetic tokens
    5. Lemmatize with POS tagging.
    """
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Replace loose punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text) # everything except letters, numbers, underscore, whitespace → space

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Filter out stopwords and any token that isn’t purely alphabetic
    stop = set(stopwords.words(language))
    filtered = [t for t in tokens if t not in stop and t.isalpha()]

    # 5. Lemmatize with POS tagging.
    pos_tags = pos_tag(filtered)

    lemmatizer = WordNetLemmatizer()
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]


    return ' '.join(lemmas)

def preprocess_text_for_wordcloud(text, language='english') -> str:
    """
    1. Lowercase
    2. Remove any character that isn’t a letter, digit or whitespace
    3. Tokenize
    4. Remove stopwords & any non-alphabetic tokens
    5. Remove words that are too frequent in all reviews to make space for more relevant words
    6. Lemmatize with POS tagging.
    """
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Replace loose punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text) # everything except letters, numbers, underscore, whitespace → space

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Filter out stopwords and any token that isn’t purely alphabetic
    stop = set(stopwords.words(language))
    
    # 6. Remove words that are too frequent in all reviews to make space for more relevant words
    stop.add("game")
    stop.add("play")
    stop.add("like")
    filtered = [t for t in tokens if t not in stop and t.isalpha()]

    # 5. Lemmatize with POS tagging.
    pos_tags = pos_tag(filtered)

    lemmatizer = WordNetLemmatizer()
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]

    return ''.join([t for t in lemmas if t not in stop]) # Refilter Stop Words after Lemmatization
    # return ' '.join(lemmas)

def get_sentiment(text: str) -> int:
    '''
    For a Text String, extract and return Sentiment Intensity as as Integer of 0 (Low Intensity) or 1 (High Intensity) 
    '''
    analyzer = SentimentIntensityAnalyzer()
    
    scores = analyzer.polarity_scores(text)

    sentiment: int = 1 if scores['pos'] > 0 else 0

    return sentiment

def is_ascii_art(text: str) -> bool:
    '''
    For a Text String, Check for high ratio of non-word characters, excessive whitespace, or repeating line patterns.
    '''
    ascii_density = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    ) / max(len(text), 1)
    return ascii_density > 0.3

def is_meme_like(text: str) -> bool:
    '''
    Returns True if the review matches a meme-like phrase pattern,
    '''
    text_lower = text.lower().strip()
    
    if len(text_lower) >= 100 or text.count("\n") >= 5:
        return False
    
    meme_patterns = [
        r"\b10/10\b(?:\W+\w+){0,8}\W+(would|will|do|recommend)",  # "10/10 would..." within 8 words
        r"^\s*(gg|ez|xd|lol)\W?$",                  # exact 1-word meme lines
        r"\bno\b.*\bu\b",                           # "no u"
        r"\bskill issue\b",                         # common meme comeback
        r"^\s*(based|cringe)\W?$",                  # culture words
        r"\+\s?rep\b",                              # reputation meme
        r"\bliterally unplayable\b",                # ironic complaint
    ]

    matches_meme_pattern = any(re.search(pat, text_lower) for pat in meme_patterns)
    return True if matches_meme_pattern else False

def is_non_english(text: str) -> bool:
    '''
    Use langdetect to determine if language is English. Treat unsure cases as non-english.
    '''
    try:
        return detect(text) != "en"
    except LangDetectException:
        return True  # Treat undetectable as suspicious

def is_too_short(text: str) -> bool:
    '''
    Checks if review is shorter than 15 characters.
    '''
    return len(text.strip()) < 15

def is_too_long(text: str) -> bool:
    '''
    Checks if review is longer than 5000 characters or has over 25 lines.
    '''
    return len(text) > 5000 or text.count("\n") > 25

def classify_suspicious_review(text: str) -> str | None:
    '''
    Checks if a review is suspicious.
    '''
    if is_too_short(text):
        return "too_short"
    elif is_too_long(text):
        return "too_long"
    elif is_ascii_art(text):
        return "ascii_art"
    elif is_meme_like(text):
        return "meme_like"
    elif is_non_english(text):
        return "non_english"
    else: return None

def get_top_words(
    texts,
    top_n: int = 100
) -> dict[str, int]:
    """
    Count word frequencies over an iterable of preprocessed text strings
    and return the top_n most common as a dict {word: count}.
    """

    counter = Counter()
    for doc in texts:
        # split on whitespace: your preprocess_text already lemmatized & removed stopwords
        tokens = doc.split()
        counter.update(tokens)
    
    if DEBUG: print(f"# get_top_words() returning {counter.most_common(top_n)}")
    return dict(counter.most_common(top_n))

def generate_wordcloud(
    frequencies: dict,
    app: apps.App,
    language: str = 'english', 
    width: int = 1200,
    height: int = 600,
    color_param: str = "hsv"
) -> None:
    """
    Generate a WordCloud image from a word frequency dict and save it to disk.
    """
    # Build Output Path
    output_path: str = f'./wordclouds/WC_{app.id}_{language}.png'
    
    # Instantiate WordCloud object
    wc = WordCloud(
    width=width,
    height=height,
    mode="RGBA",
    background_color=None, # type: ignore
    colormap=color_param
    ).generate_from_frequencies(frequencies)

    # Instantiate, set and save PLT Figure
    plt.figure(figsize=(width / 100, height / 100))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    print(f"\n> {app.name} [{app.id}] WordCloud saved to {output_path}")
    return

def save_csv(df:pd.DataFrame, file_name:str) -> bool | None:
    if any(c in file_name for c in ['.', '/']): raise Exception(f"Error: file_name '{file_name}' is invalid.")

    start = time.perf_counter()
    file_path = f"./csv/{file_name}.csv"
    
    if os.path.exists(file_path):
        print(f"⚠️  {file_path} already exists! Overwriting...")
    
    df.to_csv(
        file_path,
        index=True,
        sep=';'
    )
    
    print(f"Saved {len(df)} reviews to 'csv/{file_name}.csv' in {time.perf_counter() - start:.2f} seconds")
    return