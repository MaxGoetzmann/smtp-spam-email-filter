import argparse
import os
from email import policy
from email.parser import BytesParser

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Process .eml files in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing .eml files")
    parser.add_argument(
        "--is_useful",
        action="store_true",
        help="Indicate if the emails in this directory are useful",
    )
    return parser.parse_args()


def parse_eml(eml_path):
    with open(eml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    subject = msg["subject"]
    sender = msg["from"]
    time_sent = msg["date"]
    has_attachment = False
    attachment_types = []

    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_disposition() == "attachment":
                has_attachment = True
                attachment_types.append(part.get_content_type())

    body = clean_html(msg.get_body(preferencelist=("html", "plain")).get_content())

    return {
        utils.DDUP_FIELD_SUBJECT: subject,
        utils.DDUP_FIELD_SENDER: sender,
        utils.FIELD_TIMESTAMP: time_sent,
        utils.DDUP_FIELD_BODY: body,
        utils.FIELD_HAS_ATTACHMENT: has_attachment,
        utils.FIELD_ATTACHMENT_TYPE: attachment_types,
    }


def process_eml_directory(directory):
    eml_data = []

    dirs = os.listdir(directory)
    for filename in tqdm(dirs, total=len(dirs), desc="Parsing eml files"):
        if filename.endswith(".eml"):
            eml_path = os.path.join(directory, filename)
            email_info = parse_eml(eml_path)
            eml_data.append(email_info)

    return eml_data


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Replace tags with meaningful content
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for p in soup.find_all("p"):
        p.replace_with("\n" + p.get_text() + "\n")
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        header.replace_with("\n" + header.get_text() + "\n")
    for li in soup.find_all("li"):
        li.replace_with("\n- " + li.get_text())
    for blockquote in soup.find_all("blockquote"):
        blockquote.replace_with("\n> " + blockquote.get_text() + "\n")
    for div in soup.find_all("div"):
        div.replace_with("\n" + div.get_text() + "\n")

    return soup.get_text()


def append_to_existing_df(new_rows, old_df, order):
    new_df = pd.DataFrame(new_rows)
    out = pd.concat([old_df, new_df], ignore_index=True)
    out = out[order]
    return out


def write_eml_to_df(emails, is_useful):
    kb_model = KeyBERT()

    utils.stash_old_file(utils.METADATA_PATH)
    meta = pd.read_csv(utils.METADATA_PATH)

    utils.stash_old_file(utils.DDUP_PATH)
    ddup = pd.read_csv(utils.DDUP_PATH)

    new_meta_rows = []
    existing_emails = set()
    if utils.FIELD_EXACT_HASH in meta:
        for h in meta[utils.FIELD_EXACT_HASH]:
            existing_emails.add(h)

    new_ddup_rows = []
    existing_ddup = set()
    if utils.DDUP_FIELD_APPROX_HASH in ddup:
        for h in ddup[utils.DDUP_FIELD_APPROX_HASH]:
            existing_ddup.add(h)

    for email in tqdm(emails, total=len(emails), desc="Adding new emails to csv"):
        sender = email[utils.DDUP_FIELD_SENDER]
        subject = email[utils.DDUP_FIELD_SUBJECT]
        body = email[utils.DDUP_FIELD_BODY]
        timestamp = email[utils.FIELD_TIMESTAMP]

        exact_hash = utils.get_exact_email_hash(
            sender,
            subject,
            body,
            timestamp,
        )

        # Don't add duplicate emails to the database
        if exact_hash in existing_emails:
            continue
        existing_emails.add(exact_hash)

        # Log this exact email's arrival time and other unqiue properties.
        approx_hash = utils.lsh_by_words(sender + subject + body)

        time = utils.parse_email_timestamp(timestamp)

        new_meta_row = {
            utils.FIELD_EXACT_HASH: exact_hash,
            utils.FIELD_APPROX_HASH: approx_hash,
            utils.FIELD_HAS_ATTACHMENT: email[utils.FIELD_HAS_ATTACHMENT],
            utils.FIELD_ATTACHMENT_TYPE: email[utils.FIELD_ATTACHMENT_TYPE],
            utils.FIELD_TIMESTAMP: timestamp,
            utils.FIELD_SENDER_FREQ_MONTH: None,
            utils.FIELD_SENDER_FREQ_WEEK: None,
            utils.FIELD_SENDER_FREQ_DAY: None,
            utils.FIELD_DOM: time.day,
            utils.FIELD_DOW: time.weekday(),
            utils.FIELD_HOUR: time.hour,
            utils.FIELD_CLASSIFIER_TRUTH: is_useful,
        }
        utils.keyword_extraction(new_meta_row, kb_model)
        new_meta_rows.append(new_meta_row)

        # Check if similar emails exist in the database. If so, increment how often they occur.
        #   Otherwise, add the email contents to the database.
        if approx_hash in existing_ddup:
            row = ddup.loc[utils.DDUP_FIELD_APPROX_HASH == approx_hash]
            row[utils.DDUP_FIELD_OCCURENCES] += 1
        else:
            existing_ddup.add(approx_hash)

            polarity, subjectivity = utils.get_email_sentiment()

            new_ddup_row = {
                utils.DDUP_FIELD_APPROX_HASH: approx_hash,
                utils.DDUP_FIELD_SENDER: sender,
                utils.DDUP_FIELD_SUBJECT: subject,
                utils.DDUP_FIELD_BODY: body,
                utils.DDUP_FIELD_OCCURENCES: 1,
                utils.DDUP_FIELD_POLARITY: polarity,
                utils.DDUP_FIELD_SUBJECTIVITY: subjectivity,
                utils.DDUP_FIELD_TAGGED_USEFUL: int(is_useful),
                utils.DDUP_FIELD_TAGGED_USELESS: int(not is_useful),
                utils.DDUP_FIELD_SAID_USEFUL: [],
                utils.DDUP_FIELD_SAID_USELESS: [],
            }
            new_ddup_rows.append(new_ddup_row)

    return append_to_existing_df(
        new_meta_rows, meta, utils.FIELDS
    ), append_to_existing_df(new_ddup_rows, ddup, utils.DDUP_FIELDS)


def recalculate_aggregates():
    # For senders and recent send times

    # One hot encode file extension types

    # One hot encode most common keywords

    pass


def main():
    args = parse_args()

    emails_info = process_eml_directory(args.directory)

    # Display results
    # for email_info in emails_info:
    #     print(f"Subject: {email_info[utils.DDUP_FIELD_SUBJECT]}")
    #     print(f"Sender: {email_info[utils.DDUP_FIELD_SENDER]}")
    #     print(f"Time Sent: {email_info[utils.FIELD_TIMESTAMP]}")
    #     print(f"Has Attachment: {email_info[utils.FIELD_HAS_ATTACHMENT]}")
    #     print(
    #         f"Attachment Types: {', '.join(email_info[utils.FIELD_ATTACHMENT_TYPE]) \
    #             if email_info[utils.FIELD_ATTACHMENT_TYPE] else 'None'}"
    #     )
    #     print(f"Body:\n{email_info[utils.DDUP_FIELD_BODY]}")
    #     print("-" * 40)

    meta, ddup = write_eml_to_df(emails_info, args.is_useful)


if __name__ == "__main__":
    main()
