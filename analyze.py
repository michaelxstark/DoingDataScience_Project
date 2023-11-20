import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from textblob import TextBlob

# define file path
file_path = 'newsSpace'

# Define the column names
columns = ['source', 'url', 'title', 'image', 'category', 'description', 'rank', 'pubdate']

# Read the file line by line
with open(file_path, 'r', encoding='latin-1') as file:
    lines = file.readlines()

# Join lines
data = ''.join(lines)

#split them by separator
data = data.split(r'\N')

# now split each row to fit the columns
data = [row.strip().split('\t') for row in data]

# eliminate all rows that have missing entries.
data = [data_point for data_point in data if len(data_point) == len(columns)]

# Create the DataFrame
df = pd.DataFrame(data, columns=columns)

# adapting datatypes
df['pubdate'] = pd.to_datetime(df['pubdate'], errors='coerce')
df['rank'] = df['rank'].astype(int)

#also adapting objects to strings
string_type = columns[:-2]
df[string_type] = df[string_type].astype('string')


# a.) how is the data quality?
# b.) clustering - topics
# c.) visualize
# d.) model -- predict ranking based on datetime, source
# e.) model --  predict category
# f.) ranking -- advice company what to write about to generate good ranking.
# g.)


# cleaning the data

# ommit missing values in the pubdate

df = df.dropna(subset=['pubdate']).reset_index()


# ommit where there is no title

is_string_with_char = df['title'].apply(lambda x: isinstance(x, str) and len(x) >= 1)

# check if everything has a title
all_values_meet_condition = is_string_with_char.all()

print(all_values_meet_condition)



# image make binary - has image, has no image.

df['image'] = df['image'].apply(lambda x: 'Yes' if x != 'none' else 'No')


# ommit hours. only day month, year stay.

df['pubdate'] = df['pubdate'].dt.date


# clean the description of faulty values.

non_word_pattern = r'\W'  # This will match anything that's not a word character

# Apply a regular expression to replace non-word characters with an empty string
df['description'] = df['description'].apply(lambda x: re.sub(non_word_pattern, ' ', x))

df['description']

# perform senitment analysis and make new category.

def get_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# Apply sentiment analysis to your column and create a new column for sentiment polarity
df['sentiment_polarity'] = df['description'].apply(get_sentiment)

df

# write to csv
df.to_csv('AG_DB_Cleaned_V2.csv')

#
