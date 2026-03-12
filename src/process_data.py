import pandas as pd
import numpy as np
import os
import re

def count_entities(text):
    if not isinstance(text, str):
        return 0, 0, 0, 0
    
    length = len(text)
    hashtags = len(re.findall(r'#\w+', text))
    mentions = len(re.findall(r'@\w+', text))
    urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    
    return length, hashtags, mentions, urls

def process_tweets_chunked(file_path):
    # Process tweets in chunks to save memory
    chunk_size = 50000
    user_stats = {}
    
    print(f"   Reading tweets from {file_path}...")
    try:
        # Try-catch for potential file errors
        # Use only necessary columns: user_id, text
        # Note: In Cresci datasets, column might be 'user_id' or 'userid' or just 'id' (for the tweet).
        # Usually it is 'user_id' for the poster.
        
        # Checking columns first might be safer, but let's assume standard 'user_id' and 'text' exist or fail gracefully
        # Genuine tweets.csv headers: "id","text","source","user_id",...
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=['user_id', 'text'], dtype={'user_id': str}, encoding='utf-8', on_bad_lines='skip'):
            # Calculate stats for the chunk
            # Apply function row-wise is slow, vectorization is better
            
            # Simple vectorization
            chunk['text'] = chunk['text'].fillna('')
            chunk['tweet_len'] = chunk['text'].str.len()
            chunk['num_hashtags'] = chunk['text'].str.count('#')
            chunk['num_mentions'] = chunk['text'].str.count('@')
            chunk['num_urls'] = chunk['text'].str.count('http')
            
            # Aggregating by user_id
            grouped = chunk.groupby('user_id')[['tweet_len', 'num_hashtags', 'num_mentions', 'num_urls']].sum()
            counts = chunk.groupby('user_id').size().rename('tweet_count')
            
            # Merge into accumulating dict (simple approach for now, or just concat grouped dfs)
            # Better to concat all grouped chunks and then groupby again
            yield pd.concat([grouped, counts], axis=1)
            
    except Exception as e:
        print(f"   ⚠️ Error processing tweets file: {e}")
        return pd.DataFrame() # Return empty on failure

def load_and_process_data():
    base_path = r"e:\botDetect\data\raw\cresci-2017"
    
    # 1. Define sources
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
        
        # A. Load Users
        if not os.path.exists(user_path):
            print(f"❌ User file missing: {user_path}")
            continue
            
        try:
            users = pd.read_csv(user_path, low_memory=False, encoding='latin-1', dtype={'id': str})
        except:
            users = pd.read_csv(user_path, low_memory=False, encoding='utf-8', dtype={'id': str})
            
        users['fake'] = source['label']
        users['dataset_type'] = source['type']
        
        # B. Load & Aggregate Tweets
        if os.path.exists(tweet_path):
            chunk_aggs = []
            for agg_df in process_tweets_chunked(tweet_path):
                chunk_aggs.append(agg_df)
            
            if chunk_aggs:
                # Combine all chunk aggregations
                full_agg = pd.concat(chunk_aggs)
                # Group by user_id again (sum the sums)
                final_stats = full_agg.groupby(level=0).sum()
                
                # Calculate Averages
                final_stats['avg_tweet_len'] = final_stats['tweet_len'] / final_stats['tweet_count']
                final_stats['avg_hashtags'] = final_stats['num_hashtags'] / final_stats['tweet_count']
                final_stats['avg_mentions'] = final_stats['num_mentions'] / final_stats['tweet_count']
                final_stats['avg_urls'] = final_stats['num_urls'] / final_stats['tweet_count']
                
                # Reset index to merge
                final_stats.index.name = 'id'
                final_stats = final_stats.reset_index()
                
                # Merge with users
                print(f"   Merging user metadata with tweet stats...")
                users = users.merge(final_stats[['id', 'avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls']], on='id', how='left')
                
                # Fill missing (users who didn't tweet) with 0
                users[['avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls']] = users[['avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls']].fillna(0)
            else:
                 print("   ⚠️ No tweets read.")
        else:
            print(f"   ⚠️ Tweets file missing: {tweet_path}")
            # Add columns with 0s if missing
            users['avg_tweet_len'] = 0
            users['avg_hashtags'] = 0
            users['avg_mentions'] = 0
            users['avg_urls'] = 0

        all_user_dfs.append(users)

    # Combine all
    full_df = pd.concat(all_user_dfs, ignore_index=True)
    print(f"\nTotal Combined Records: {len(full_df)}")

    # Feature Engineering
    features_to_keep = [
        'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'geo_enabled', 'default_profile', 'profile_use_background_image', 'verified', 'protected',
        'avg_tweet_len', 'avg_hashtags', 'avg_mentions', 'avg_urls',
        'fake'
    ]
    
    # Ensure binary columns
    binary_cols = ['geo_enabled', 'default_profile', 'profile_use_background_image', 'verified', 'protected']
    for col in binary_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0).astype(int)

    # Fill numerical NaNs
    full_df = full_df.fillna(0)
    
    # Derived Feature: Reputation
    full_df['reputation'] = full_df['followers_count'] / (full_df['followers_count'] + full_df['friends_count'] + 1)
    
    # Final Selection
    processed_df = full_df[features_to_keep + ['reputation']].copy()
    
    # Shuffle
    processed_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    processed_df.to_csv(output_path, index=False)
    print(f"\n✅ Processed data saved to {output_path}")
    print(processed_df.head())

if __name__ == "__main__":
    load_and_process_data()