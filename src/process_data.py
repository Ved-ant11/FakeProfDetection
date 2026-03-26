import pandas as pd
import numpy as np
import os
import re
import math
from collections import Counter


def username_entropy(name):
    if not isinstance(name, str) or len(name) == 0:
        return 0.0
    counts = Counter(name)
    length = len(name)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def count_entities(text):
    if not isinstance(text, str):
        return 0, 0, 0, 0
    length = len(text)
    hashtags = len(re.findall(r'#\w+', text))
    mentions = len(re.findall(r'@\w+', text))
    urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    return length, hashtags, mentions, urls


def process_tweets_chunked(file_path):
    chunk_size = 50000

    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=['user_id', 'text'], dtype={'user_id': str}, encoding='utf-8', on_bad_lines='skip'):
            chunk['text'] = chunk['text'].fillna('')
            chunk['tweet_len'] = chunk['text'].str.len()
            chunk['num_hashtags'] = chunk['text'].str.count('#')
            chunk['num_mentions'] = chunk['text'].str.count('@')
            chunk['num_urls'] = chunk['text'].str.count('http')

            grouped = chunk.groupby('user_id')[['tweet_len', 'num_hashtags', 'num_mentions', 'num_urls']].sum()
            counts = chunk.groupby('user_id').size().rename('tweet_count')

            yield pd.concat([grouped, counts], axis=1)

    except Exception as e:
        print(f"   Warning: Error processing tweets file: {e}")
        return pd.DataFrame()


def load_and_process_data():
    base_path = r"e:\botDetect\data\raw\cresci-2017"

    sources = [
        {"path": os.path.join(base_path, "genuine", "genuine_accounts.csv"), "type": "Genuine", "label": 0},
        {"path": os.path.join(base_path, "spambots1", "social_spambots_1.csv"), "type": "Social Spambots 1", "label": 1},
        {"path": os.path.join(base_path, "spambots2", "social_spambots_2.csv"), "type": "Social Spambots 2", "label": 1},
        {"path": os.path.join(base_path, "spambots3", "social_spambots_3.csv"), "type": "Social Spambots 3", "label": 1},
        {"path": os.path.join(base_path, "traditional1", "traditional_spambots_1.csv"), "type": "Traditional Spambots 1", "label": 1},
    ]

    output_dir = r"e:\botDetect\data\processed"
    output_path = os.path.join(output_dir, "cresci_expanded_with_content.csv")

    all_user_dfs = []

    for source in sources:
        user_path = os.path.join(source["path"], "users.csv")
        tweet_path = os.path.join(source["path"], "tweets.csv")

        print(f"Processing {source['type']}...")

        if not os.path.exists(user_path):
            print(f"User file missing: {user_path}")
            continue

        try:
            users = pd.read_csv(user_path, low_memory=False, encoding='latin-1', dtype={'id': str})
        except Exception:
            users = pd.read_csv(user_path, low_memory=False, encoding='utf-8', dtype={'id': str})

        users['fake'] = source['label']
        users['dataset_type'] = source['type']

        if 'screen_name' in users.columns:
            users['username_entropy'] = users['screen_name'].apply(username_entropy)
        else:
            users['username_entropy'] = 0.0

        if os.path.exists(tweet_path):
            chunk_aggs = []
            for agg_df in process_tweets_chunked(tweet_path):
                chunk_aggs.append(agg_df)

            if chunk_aggs:
                full_agg = pd.concat(chunk_aggs)
                final_stats = full_agg.groupby(level=0).sum()

                final_stats['avg_tweet_len'] = final_stats['tweet_len'] / final_stats['tweet_count']
                final_stats['avg_hashtags'] = final_stats['num_hashtags'] / final_stats['tweet_count']
                final_stats['avg_mentions'] = final_stats['num_mentions'] / final_stats['tweet_count']
                final_stats['avg_urls'] = final_stats['num_urls'] / final_stats['tweet_count']

                final_stats.index.name = 'id'
                final_stats = final_stats.reset_index()

                print(f"   Merging user metadata with tweet stats...")
                users = users.merge(final_stats[['id', 'avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls']], on='id', how='left')

                users[['avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls']] = users[['avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls']].fillna(0)
            else:
                print("   No tweets read.")
        else:
            print(f"   Tweets file missing: {tweet_path}")
            users['avg_tweet_len'] = 0
            users['avg_hashtags'] = 0
            users['avg_mentions'] = 0
            users['avg_urls'] = 0

        all_user_dfs.append(users)

    full_df = pd.concat(all_user_dfs, ignore_index=True)
    print(f"\nTotal Combined Records: {len(full_df)}")

    features_to_keep = [
        'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'geo_enabled', 'default_profile', 'profile_use_background_image', 'verified', 'protected',
        'avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls',
        'username_entropy',
        'fake'
    ]

    binary_cols = ['geo_enabled', 'default_profile', 'profile_use_background_image', 'verified', 'protected']
    for col in binary_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0).astype(int)

    full_df = full_df.fillna(0)

    full_df['reputation'] = full_df['followers_count'] / (full_df['followers_count'] + full_df['friends_count'] + 1)

    processed_df = full_df[features_to_keep + ['reputation']].copy()

    processed_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    print(processed_df.head())


if __name__ == "__main__":
    load_and_process_data()