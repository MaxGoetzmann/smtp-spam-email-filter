import hashlib
from math import ceil
from pickle import dumps


def object_to_bytes(obj):
    """
    Convert any Python object to a sequence of bytes.
    Identical objects (with the same field values) produce the same byte sequence.
    """
    # Handle custom objects by converting them to a consistent representation
    if hasattr(obj, "__dict__"):
        # Sort the dictionary to ensure consistent ordering
        obj_dict = {k: object_to_bytes(v) for k, v in sorted(obj.__dict__.items())}
        return dumps(obj_dict)

    # For other objects, just use pickle
    return dumps(obj)


def hash_text_by_words(text, windows=10, overlap_pct=0.5):
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


def combine_hashes(hashes):
    """
    Combine a list of hash values into a single hash.

    :param hashes: A list of hash values.
    :return: A combined hash value.
    """
    true_max = "0" * 56  # Alphanumeric lowest value for SHA224 (longest hash)
    for i in range(len(hashes[0])):
        hash_type_max = [segment[i] for segment in hashes]
        true_max = max(true_max, hash_type_max)
        print(i, true_max)

    return hashlib.md5(true_max.encode()).hexdigest()


def lsh_by_words(text, windows=10, overlap_pct=0.5):
    """
    Perform locality-sensitive hashing on the input text by words.

    :param text: The input text to hash.
    :param window_size: The size of the sliding window (in number of words).
    :return: A single hash value representing the LSH.
    """
    hashes = hash_text_by_words(text, windows, overlap_pct)
    return combine_hashes(hashes)
