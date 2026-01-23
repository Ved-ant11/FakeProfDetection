import pandas as pd
import numpy as np
from faker import Faker
import random
import os

def generate_synthetic_data(num_samples=1000):
    """
    Generates a synthetic dataset for fake profile detection.
    
    Features based on common research indicators:
    - profile_pic: 1 if present
    - nums_length_username: ratio of numerical chars in username
    - fullname_words: number of words in full name
    - nums_length_fullname: ratio of numerical chars in full name
    - name_username_match: 1 if similar
    - description_length: length of bio
    - external_url: 1 if present
    - private: 1 if profile is private
    - posts: number of posts
    - followers: number of followers
    - follows: number of accounts followed
    - fake: 0 (Real) or 1 (Fake)
    """
    fake = Faker()
    data = []

    # Generate 'Real' accounts (Target = 0)
    for _ in range(num_samples // 2):
        profile = {}
        profile['profile_pic'] = 1  
        
        # Real usernames usually make sense
        username = fake.user_name()
        profile['nums_length_username'] = sum(c.isdigit() for c in username) / len(username) if len(username) > 0 else 0
        
        fullname = fake.name()
        profile['fullname_words'] = len(fullname.split())
        profile['nums_length_fullname'] = sum(c.isdigit() for c in fullname) / len(fullname) if len(fullname) > 0 else 0
        
        # Real users often have matching names/usernames
        profile['name_username_match'] = 1 if fullname.split()[0].lower() in username.lower() else 0
        
        profile['description_length'] = len(fake.text(max_nb_chars=100))
        profile['external_url'] = random.choice([0, 1])
        profile['private'] = random.choice([0, 1])
        
        # Real user engagement patterns
        profile['posts'] = int(np.random.normal(100, 50))
        if profile['posts'] < 0: profile['posts'] = 0
        
        profile['followers'] = int(np.random.exponential(500))
        profile['follows'] = int(np.random.normal(300, 100))
        if profile['follows'] < 0: profile['follows'] = 0
        
        profile['fake'] = 0
        data.append(profile)

    # Generate 'Fake' accounts (Target = 1)
    for _ in range(num_samples // 2):
        profile = {}
        # Fake accounts often lack profile pics
        profile['profile_pic'] = random.choices([0, 1], weights=[0.7, 0.3])[0]
        
        # Fake usernames often random/numeric
        username = fake.lexify(text='????????') + str(random.randint(1000, 9999))
        profile['nums_length_username'] = sum(c.isdigit() for c in username) / len(username)
        
        fullname = fake.name()
        if random.random() > 0.8: fullname = "" # Sometimes no name
        profile['fullname_words'] = len(fullname.split())
        profile['nums_length_fullname'] = 0
        
        profile['name_username_match'] = 0
        
        profile['description_length'] = 0 if random.random() > 0.5 else len(fake.text(max_nb_chars=20))
        profile['external_url'] = random.choices([0, 1], weights=[0.8, 0.2])[0]
        profile['private'] = random.choices([0, 1], weights=[0.9, 0.1])[0] 
        
        # Fake engagement patterns
        profile['posts'] = int(np.random.exponential(5))
        profile['followers'] = int(np.random.exponential(20)) 
        profile['follows'] = int(np.random.normal(1000, 300)) 
        if profile['follows'] < 0: profile['follows'] = 0
        
        profile['fake'] = 1
        data.append(profile)

    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} samples saved to {output_path}")
    
    # Stats
    print("\nDataset Balance:")
    print(df['fake'].value_counts())

if __name__ == "__main__":
    generate_synthetic_data(2000)
