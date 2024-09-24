import hashlib
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import ceil

from keybert import KeyBERT
from pandas import DataFrame, concat, get_dummies
from textblob import TextBlob
from tqdm import tqdm

# csv outputs
METADATA_PATH = "email_metadata.csv"
DDUP_PATH = "email_content.csv"

# XGBoost hyperparameters
MANUAL_TAG_WEIGHT = 1

# Clustering hyperparameters
NN_REDUCTION_DIM = 50
CLUSTER_EPOCHS = 200
CLUSTER_BATCH_SIZE = 32
K_MEANS_CT = 4

# Keyword finding arguments
KW_DIM = (1, 2)
KW_DIVERSITY = 0.7
EMBED_MOST_FREQUENT = 1000

# Adjust number of keywords to extract from each email part
SUBJECT_KW_CT = 5
BODY_KW_CT = 10

# ddup table fields
DDUP_FIELD_APPROX_HASH = "Approximate Email Hash"
DDUP_FIELD_SENDER = "Sender"
DDUP_FIELD_SUBJECT = "Subject"
DDUP_FIELD_BODY = "Body"
DDUP_FIELD_OCCURENCES = "Occurences"
DDUP_FIELD_POLARITY = "Polarity"  # sentiment analysis
DDUP_FIELD_SUBJECTIVITY = "Subjectivity"  # sentiment analysis
DDUP_FIELD_TAGGED_USEFUL = "Tagged Useful Alert"
DDUP_FIELD_TAGGED_USELESS = "Tagged Useless Alert"
DDUP_FIELD_SAID_USEFUL = "People Who Said Useful"
DDUP_FIELD_SAID_USELESS = (
    "People Who Said Useless"  # Trumps appearences in "Useful" list
)

DDUP_FIELDS = [
    DDUP_FIELD_APPROX_HASH,
    DDUP_FIELD_SENDER,
    DDUP_FIELD_SUBJECT,
    DDUP_FIELD_BODY,
    DDUP_FIELD_OCCURENCES,
    DDUP_FIELD_POLARITY,
    DDUP_FIELD_SUBJECTIVITY,
    DDUP_FIELD_TAGGED_USEFUL,
    DDUP_FIELD_TAGGED_USELESS,
    DDUP_FIELD_SAID_USEFUL,
    DDUP_FIELD_SAID_USELESS,
]

DDUP_FIELD_SUBJECT_KWS = "Subject Keyword"
DDUP_FIELD_BODY_KWS = "Body Keyword"

# Precompute keyword fields to avoid redundant calculations
SUBJECT_KEYWORD_FIELDS = [f"{DDUP_FIELD_SUBJECT_KWS} {i}" for i in range(SUBJECT_KW_CT)]
BODY_KEYWORD_FIELDS = [f"{DDUP_FIELD_BODY_KWS} {i}" for i in range(BODY_KW_CT)]

# Append keyword fields to DDUP_FIELDS
DDUP_FIELDS.extend(SUBJECT_KEYWORD_FIELDS)
DDUP_FIELDS.extend(BODY_KEYWORD_FIELDS)


def serve_subject_keyword_fields():
    return SUBJECT_KEYWORD_FIELDS


def serve_body_keyword_fields():
    return BODY_KEYWORD_FIELDS


def repeat_for_subject_and_body(func, *args, **kwargs):
    for i in SUBJECT_KEYWORD_FIELDS:
        func(i, *args, **kwargs)

    for i in BODY_KEYWORD_FIELDS:
        func(i, *args, **kwargs)


# Temporal email constants
MONTH_LENGTH_HR = 35 * 24 - 1  # catch emails sent on the 1st Monday of every month
WEEK_LENGTH_HR = 7 * 24 - 1
DAY_LENGTH_HR = 24 - 1

# Set pandas column headers
FIELD_EXACT_HASH = "Exact Email Hash"
FIELD_APPROX_HASH = "Approximate Email Hash"
FIELD_TIMESTAMP = "Arrival Timestamp"
FIELD_HAS_ATTACHMENT = "Has Attachment"
FIELD_ATTACHMENT_TYPE = "Attatchment Type"
FIELD_SENDER_FREQ_MONTH = "Sender Frequency for Month"
FIELD_SENDER_FREQ_WEEK = "Sender Frequency for Week"
FIELD_SENDER_FREQ_DAY = "Sender Frequency for Day"
FIELD_DOM = "Arrival Day of Month"
FIELD_DOW = "Arrival Day of Week"
FIELD_HOUR = "Arrival Hour"
FIELD_CLASSIFIER_TRUTH = "Classifier Truth"

FIELDS = [
    FIELD_EXACT_HASH,
    FIELD_APPROX_HASH,
    FIELD_HAS_ATTACHMENT,
    FIELD_ATTACHMENT_TYPE,
    FIELD_TIMESTAMP,
    FIELD_SENDER_FREQ_MONTH,
    FIELD_SENDER_FREQ_WEEK,
    FIELD_SENDER_FREQ_DAY,
    FIELD_DOM,
    FIELD_DOW,
    FIELD_HOUR,
    FIELD_CLASSIFIER_TRUTH,
]


def _object_to_bytes(obj):
    """
    Convert any Python object to a sequence of bytes.
    Identical objects (with the same field values) produce the same byte sequence.
    """
    # Handle custom objects by converting them to a consistent representation
    if hasattr(obj, "__dict__"):
        # Sort the dictionary to ensure consistent ordering
        obj_dict = {k: _object_to_bytes(v) for k, v in sorted(obj.__dict__.items())}
        return pickle.dumps(obj_dict)

    # For other objects, just use pickle
    return pickle.dumps(obj)


def get_exact_email_hash(sender, subject, body, time):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Update the hash object with the bytes of the input string
    s = (
        _object_to_bytes(sender)
        + _object_to_bytes(subject)
        + _object_to_bytes(body)
        + _object_to_bytes(time)
    )
    md5_hash.update(s.encode("utf-8"))

    # Return the hexadecimal representation of the digest
    return md5_hash.hexdigest()


def output_csv(path: str, df: DataFrame):
    df.to_csv(path, index=False, encoding="utf-8")


# Temporary LSH before I create the full library
def _hash_text_by_words(text, windows=10, overlap_pct=0.5):
    """
    Hashes a text using a sliding window over words with overlap.

    :param text: The input text to hash.
    :param window_size: The size of the sliding window (in number of words).
    :param overlap: The number of overlapping words between windows.
    :return: A list of hash values for each window of words in the text.
    """
    words = text.split()
    total_words = len(words)

    window_size = ceil(total_words / windows)
    overlap = ceil(window_size * overlap_pct)

    # Calculate the number of windows
    if total_words < window_size:
        window_size = overlap + 1
    num_windows = (total_words - window_size) // (window_size - overlap) + 1

    hash_list = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        if end > total_words:
            end = total_words
            start = max(0, end - window_size)

        window = " ".join(words[start:end])

        # 3 fastest python hashes
        md5 = hashlib.md5(window.encode()).hexdigest()
        sha1 = hashlib.sha1(window.encode()).hexdigest()
        sha224 = hashlib.sha224(window.encode()).hexdigest()

        hash_list.append((md5, sha1, sha224))

    return hash_list


def _combine_hashes(hashes):
    """
    Combine a list of hash values into a single hash.

    :param hashes: A list of hash values.
    :return: A combined hash value.
    """
    true_max = "0" * 56  # Alphanumeric lowest value for SHA224 (longest hash)
    for i in range(len(hashes[0])):
        hash_type_max = []
        for j in range(len(hashes)):
            hash_type_max.append(hashes[j][i])
        true_max = max(true_max, max(hash_type_max))
        print(i, true_max)

    return hashlib.md5(true_max.encode()).hexdigest()


def lsh_by_words(text, windows=10, overlap_pct=0.5):
    """
    Perform locality-sensitive hashing on the input text by words.

    :param text: The input text to hash.
    :param window_size: The size of the sliding window (in number of words).
    :return: A single hash value representing the LSH.
    """
    hashes = _hash_text_by_words(text, windows, overlap_pct)
    return _combine_hashes(hashes)


def stash_old_file(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    # Create the new file path by appending '.old' before the file extension
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}.old{ext}"

    # Copy the file to the new path
    try:
        shutil.copy(file_path, new_file_path)
        print(f"File duplicated successfully as {new_file_path}.")
    except IOError as e:
        print(f"Error copying file: {e}")


def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def get_keywords(text):
    pass


def parse_email_timestamp(time: str) -> datetime:
    return datetime.strptime(time, "%a, %d %b %Y %H:%M:%S %z")


def one_hot_encode_column(df, column_name):
    """
    One-hot encodes a specified column in a DataFrame where each cell
    contains an array of strings and appends the one-hot encoded columns
    to the original DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to encode.
    - column_name (str): The name of the column to one-hot encode.

    Returns:
    - pd.DataFrame: The DataFrame with the one-hot encoded columns appended.
    """
    # Ensure the specified column is in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Flatten the arrays of strings to individual strings
    flattened = df[column_name].explode()

    # One-hot encode the flattened data
    one_hot = get_dummies(flattened)

    # Group by the original index and sum to get the one-hot encoding back to the original shape
    one_hot = one_hot.groupby(level=0).sum()

    # Concatenate the one-hot encoded columns with the original DataFrame
    df_encoded = concat([df.drop(columns=[column_name]), one_hot], axis=1)

    return df_encoded


def keyword_extraction(df, kw_model: KeyBERT, diversity=KW_DIVERSITY, dims=KW_DIM):
    def keyword_extraction_helper(text, kw_ct, field_iter, index):
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=dims,
            top_n=kw_ct,
            stop_words=None,
            use_mmr=True,
            diversity=diversity,
        )
        kw_base = 0
        for j in field_iter:
            if kw_base < len(keywords):
                df.loc[index, j] = keywords[kw_base][0]
            kw_base += 1

    def process_row(index, row):
        keyword_extraction_helper(
            row[DDUP_FIELD_SUBJECT],
            SUBJECT_KW_CT,
            subject_iter,
            index,
        )
        keyword_extraction_helper(row[DDUP_FIELD_BODY], BODY_KW_CT, body_iter, index)

    subject_iter = serve_subject_keyword_fields()
    body_iter = serve_body_keyword_fields()

    with ThreadPoolExecutor() as executor:
        # Submit tasks for each row to the executor
        futures = {executor.submit(process_row, i, row): i for i, row in df.iterrows()}

        # Wait for all futures to complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting keywords"
        ):
            try:
                future.result()
            except:
                pass
