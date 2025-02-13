import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from math import ceil

import pandas as pd
import settings
from keybert import KeyBERT
from numpy import nan
from tqdm import tqdm


def generate_clustered_hours(
    n,
    start_date="2022-01-01 00:00",
    end_date="2023-12-31 23:59",
    cluster_count=50,
    cluster_duration_hours=6,
    cluster_probability=0.5,
):
    start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
    total_hours = int((end_date - start_date).total_seconds() / 3600)

    clusters = [
        (
            start_date
            + timedelta(hours=random.randint(0, total_hours - cluster_duration_hours)),
            cluster_duration_hours,
        )
        for _ in range(cluster_count)
    ]

    def random_hour():
        return start_date + timedelta(hours=random.randint(0, total_hours - 1))

    hours = []
    for _ in range(n):
        if random.random() < cluster_probability:
            cluster_start, duration = random.choice(clusters)
            hour_within_cluster = cluster_start + timedelta(
                hours=random.randint(0, duration - 1)
            )
            hours.append(
                (hour_within_cluster.timestamp() // 3600, hour_within_cluster.hour)
            )
        else:
            rand_time = random_hour()
            hours.append((rand_time.timestamp() // 3600, rand_time.hour))

    return hours


def alter_string(input_string, is_useless, chance=0.5, max_seeds=10, iters=10):
    if random.random() > chance:
        return input_string, is_useless

    random.seed(random.randint(1, max_seeds))
    words = input_string.split()
    additions = [
        "quickly",
        "really",
        "extremely",
        "very",
        "quite",
        "simply",
        "absolutely",
        "actually",
        "totally",
        "slightly",
    ]
    synonyms = {
        "good": ["great", "excellent", "fine", "superb", "wonderful"],
        "bad": ["awful", "terrible", "poor", "dreadful", "horrible"],
        "happy": ["joyful", "content", "pleased", "cheerful", "delighted"],
        "sad": ["unhappy", "down", "miserable", "depressed", "melancholy"],
        "fast": ["quick", "speedy", "swift", "rapid", "brisk"],
        "slow": ["lethargic", "sluggish", "unhurried", "lazy", "gradual"],
        "big": ["large", "huge", "massive", "gigantic", "enormous"],
        "small": ["tiny", "little", "miniature", "compact", "petite"],
        "hot": ["warm", "boiling", "scorching", "sweltering", "toasty"],
        "cold": ["cool", "chilly", "freezing", "frigid", "icy"],
    }
    opposites = {
        "good": "bad",
        "bad": "good",
        "happy": "sad",
        "sad": "happy",
        "fast": "slow",
        "slow": "fast",
        "big": "small",
        "small": "big",
        "hot": "cold",
        "cold": "hot",
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
        "in": "out",
        "out": "in",
    }

    for _ in range(iters):
        if not words:
            break
        action = random.randint(0, 2)
        if action == 0:
            index = random.randint(0, len(words))
            words.insert(index, random.choice(additions))
        elif action == 1:
            word = random.choice(words)
            if word in synonyms:
                words[words.index(word)] = random.choice(synonyms[word])
        elif action == 2:
            word = random.choice(words)
            if word in opposites:
                words[words.index(word)] = opposites[word]

            # Reverse spam intentions
            if is_useless:
                is_useless = 0
            else:
                is_useless = 1
            break

    return " ".join(words), is_useless


def email_freq_distribution(
    num_options, range1_bounds=(0, 4), range2_bounds=(20, 51), range3_bounds=(300, 1001)
):
    # Define the ranges and their corresponding proportions
    range1 = list(range(range1_bounds[0], range1_bounds[1]))
    range2 = list(range(range2_bounds[0], range2_bounds[1]))
    range3 = list(range(range3_bounds[0], range3_bounds[1]))

    # Calculate the number of elements for each range based on the proportions
    count_range1 = int(0.5 * num_options)  # 50% of 100
    count_range2 = int(0.25 * num_options)  # 25% of 100
    count_range3 = num_options - count_range1 - count_range2  # The rest

    # Generate the numbers
    numbers_range1 = random.choices(range1, k=count_range1)
    numbers_range2 = random.choices(range2, k=count_range2)
    numbers_range3 = random.choices(range3, k=count_range3)

    # Combine the numbers
    return numbers_range1 + numbers_range2 + numbers_range3


def generate_random_email_sender():
    return random.choice(settings.EMAIL_ADDR_EXAMPLES)


def separate_email(email_content):
    clean = email_content.replace("Subject: ", "", 1)
    parts = clean.split("\n", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def count_email_occurrences(email_time, data_dict):
    # Initialize counters for occurrences
    week_occurrences = 0
    month_occurrences = 0

    # Calculate the start times in hours since the Unix epoch
    week_start = email_time - settings.WEEK_LENGTH
    month_start = email_time - settings.MONTH_LENGTH

    # Iterate over the data_dict and count occurrences
    for hour, occurrences in data_dict.items():
        if week_start <= hour <= email_time:
            week_occurrences += occurrences
        if month_start <= hour <= email_time:
            month_occurrences += occurrences

    return week_occurrences, month_occurrences


def email_df_generation(df):
    out_rows = []
    ddup_rows = []
    ddup_hashes = set()
    distro = email_freq_distribution(100, range3_bounds=(51, 100))

    count = 0
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Generating initial email dataframe"
    ):
        sender = generate_random_email_sender()
        is_useless = int(row["label_num"])

        copies = random.choice(distro)
        count += copies
        for _ in range(copies):
            subject, body = separate_email(row["text"])
            subject, is_useless_subject = alter_string(
                subject, is_useless, max_seeds=3, chance=0.3, iters=3
            )
            body, is_useless_body = alter_string(
                body, is_useless, max_seeds=5, chance=0.4, iters=len(body) // 4
            )
            is_useless = int(not (not is_useless_subject or not is_useless_body))

            hash = settings.get_email_hash(sender, subject, body)

            if hash in ddup_hashes:
                # Update existing hash
                for ddup_row in ddup_rows:
                    if ddup_row[settings.DDUP_FIELD_HASH] == hash:
                        ddup_row[settings.DDUP_FIELD_OCCURENCES] += 1
                        break
            else:
                # Add new hash
                ddup_rows.append(
                    {
                        settings.DDUP_FIELD_HASH: hash,
                        settings.DDUP_FIELD_SUBJECT: subject,
                        settings.DDUP_FIELD_BODY: body,
                        settings.DDUP_FIELD_OCCURENCES: 1,
                    }
                )

                ddup_hashes.add(hash)

            useful_ct = 0
            useless_ct = 0
            if is_useless:
                useful_ct = generate_use_number(ceil(useless_ct / 2))
                useless_ct = generate_use_number(5)
            else:
                useful_ct = generate_use_number(5)
                useless_ct = generate_use_number(ceil(useful_ct / 2))

            out_rows.append(
                {
                    settings.FIELD_HASH: hash,
                    settings.FIELD_ATTACHMENT: random.randint(1, 500) == 1,
                    settings.FIELD_SENDER: sender,
                    settings.FIELD_SENDER_FREQ_MONTH: None,
                    settings.FIELD_SENDER_FREQ_WEEK: None,
                    settings.FIELD_SENDER_FREQ_DAY: None,
                    settings.FIELD_HOUR: None,
                    settings.FIELD_TAGGED_USEFUL: useful_ct,
                    settings.FIELD_TAGGED_USELESS: useless_ct,
                    settings.FIELD_CLASSIFIER_TRUTH: is_useless,
                }
            )

    out = pd.DataFrame(out_rows, columns=settings.FIELDS)
    ddup = pd.DataFrame(ddup_rows, columns=settings.DDUP_FIELDS)
    return out, ddup, count


def sender_df_generation(df, count):
    senders_dict = {}
    hours = sorted(generate_clustered_hours(count))

    for i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating sender dataframe and temporal data",
    ):
        abs_hour = hours[i][0]
        day_hour = hours[i][1]
        sender = row[settings.FIELD_SENDER]

        if sender not in senders_dict:
            senders_dict[sender] = {}

        hour_dict = senders_dict[sender]
        if abs_hour in hour_dict:
            hour_dict[abs_hour] += 1
        else:
            hour_dict[abs_hour] = 1

        df.loc[i, settings.FIELD_HOUR] = day_hour
        day_occ = hour_dict.get(abs_hour, 0)
        week_occ, month_occ = count_email_occurrences(abs_hour, hour_dict)
        df.loc[i, settings.FIELD_SENDER_FREQ_DAY] = day_occ
        df.loc[i, settings.FIELD_SENDER_FREQ_WEEK] = week_occ
        df.loc[i, settings.FIELD_SENDER_FREQ_MONTH] = month_occ

    senders = pd.DataFrame(
        [
            {settings.SENDER_FIELD_NAME: sender, settings.SENDER_FIELD_DATES: dates}
            for sender, dates in senders_dict.items()
        ]
    )

    return df, senders


def generate_use_number(max, dec=2):
    choices = list(range(0, max + 1))
    remainder = 1
    weights = []
    for _ in range(max + 1):
        weights.append(remainder)
        remainder /= dec
    return random.choices(choices, weights=weights)[0]


def keyword_extraction(emails, kw_model: KeyBERT, diversity=0.7):
    def keyword_extraction_helper(part, kw_ct, serve_field_iter):
        keywords = kw_model.extract_keywords(
            row[part],
            keyphrase_ngram_range=(1, 2),
            top_n=kw_ct,
            stop_words=None,
            use_mmr=True,
            diversity=diversity,
        )
        kw_base = 0
        for j in serve_field_iter:
            emails.loc[i, j] = keywords[kw_base]
            kw_base += 1

    for i, row in emails.iterrows():
        keyword_extraction_helper(
            settings.DDUP_FIELD_SUBJECT,
            settings.SUBJECT_KW_CT,
            settings.serve_subject_keyword_fields(),
        )
        keyword_extraction_helper(
            settings.DDUP_FIELD_BODY,
            settings.BODY_KW_CT,
            settings.serve_body_keyword_fields(),
        )


def keyword_extraction(df, kw_model: KeyBERT, diversity=0.7):
    def keyword_extraction_helper(text, kw_ct, field_iter, index):
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=settings.KW_DIM,
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
            row[settings.DDUP_FIELD_SUBJECT],
            settings.SUBJECT_KW_CT,
            subject_iter,
            index,
        )
        keyword_extraction_helper(
            row[settings.DDUP_FIELD_BODY], settings.BODY_KW_CT, body_iter, index
        )

    subject_iter = settings.serve_subject_keyword_fields()
    body_iter = settings.serve_body_keyword_fields()

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


def final_df_merge(emails, email_kws):
    def add_keyword_email_fields(field, emails):
        emails[field] = nan

    # Add new fields
    add_keyword_email_fields(settings.DDUP_FIELD_OCCURENCES, emails)
    settings.repeat_for_subject_and_body(add_keyword_email_fields, emails)

    # Get keywords of corresponding hash
    def get_keyword_by_hash(field, email_row_idx, ddup_row):
        emails.loc[email_row_idx, field] = ddup_row[field]

    for i, row in tqdm(
        emails.iterrows(),
        total=len(emails),
        desc="Pulling keywords from DDUP dataframe",
    ):
        hash = row[settings.FIELD_HASH]
        ddup_row = email_kws.loc[email_kws[settings.DDUP_FIELD_HASH] == hash]

        if not ddup_row.empty:
            ddup_row = ddup_row.iloc[0]
        else:
            raise Exception(
                "No matching hash found in the DDUP table. What did you do!?"
            )

        settings.repeat_for_subject_and_body(get_keyword_by_hash, i, ddup_row)

        emails.loc[i, settings.DDUP_FIELD_OCCURENCES] = ddup_row[
            settings.DDUP_FIELD_OCCURENCES
        ]

    # Get embedding of keywords
    def get_embedding_field(s):
        return f"embedding: {s}"

    def add_unique_keyword(field, row_idx, kw_counter):
        keyword = emails.loc[row_idx, field]
        kw_counter[keyword] += 1

    # Create a Counter object to count keyword frequencies
    kw_counter = Counter()
    for i, row in tqdm(
        emails.iterrows(),
        total=len(emails),
        desc="Counting keyword frequencies",
    ):
        settings.repeat_for_subject_and_body(add_unique_keyword, i, kw_counter)

    # Get the n most common keywords
    most_common_keywords = [
        kw for kw, _ in kw_counter.most_common(settings.EMBED_MOST_FREQUENT)
    ]

    # To improve performance, collect all the new columns and then concatenate them to the DataFrame at once
    new_columns = {}
    for kw in most_common_keywords:
        new_columns[get_embedding_field(kw)] = False

    new_columns_df = pd.DataFrame(new_columns, index=emails.index)
    emails = pd.concat([emails, new_columns_df], axis=1)

    # Set the values to True for the corresponding keywords in the DataFrame
    def add_embedding_to_row(field, row_idx):
        embedding_field = get_embedding_field(emails.loc[row_idx, field])
        if embedding_field in emails.columns:
            emails.loc[row_idx, embedding_field] = True

    for i, row in tqdm(
        emails.iterrows(),
        total=len(emails),
        desc="Adding embeddings to DataFrame",
    ):
        settings.repeat_for_subject_and_body(add_embedding_to_row, i)

    # Finally write the keywords to their new columns
    for i, row in tqdm(
        emails.iterrows(),
        total=len(emails),
        desc="Populating embedding",
    ):
        settings.repeat_for_subject_and_body(add_embedding_to_row, i)

    return emails


def main():
    # df = pd.read_csv("spam_ham_dataset.csv")
    # step1, ddup, count = email_df_generation(df)
    # emails, senders = sender_df_generation(step1, count)

    # settings.output_csv(settings.DDUP_CSV_PATH, ddup)
    # settings.output_csv(settings.SENDER_CSV_PATH, senders)
    # settings.output_csv(settings.EMAIL_CSV_PATH, emails)

    ### TEMP ###
    emails = pd.read_csv(settings.EMAIL_CSV_PATH)
    ddup = pd.read_csv(settings.DDUP_CSV_PATH)
    ############

    # kw_model = KeyBERT()
    # keyword_extraction(ddup, kw_model, diversity=0.7)
    # settings.output_csv(settings.FINISHED_DDUP_PATH, ddup)

    ### TEMP ###
    ddup = pd.read_csv(settings.FINISHED_DDUP_PATH)
    ############

    final = final_df_merge(emails, ddup)
    settings.output_csv(settings.FINISHED_DF, final)


if __name__ == "__main__":
    main()
