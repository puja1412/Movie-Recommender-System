#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# In[3]:


movies.head(1)

# In[4]:


credits.head(1)

# In[5]:


movies.shape

# In[6]:


credits.shape

# In[7]:


movies.merge(credits, on='title').shape

# In[8]:


movies = movies.merge(credits, on='title')

# In[9]:


movies.head(1)

# In[10]:


movies['original_language'].value_counts()

# In[11]:


movies.info()

# In[12]:


movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# In[13]:


movies.head()

# In[14]:


movies.isnull().sum()

# In[15]:


movies.dropna(inplace=True)

# In[16]:


movies.duplicated().sum()

# In[17]:


movies.iloc[0].genres


# In[18]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[19]:


import ast

ast.literal_eval(
    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

# In[20]:


movies['genres'] = movies['genres'].apply(convert)

# In[21]:


movies.head()

# In[22]:


movies['keywords'] = movies['keywords'].apply(convert)

# In[23]:


movies.head()


# In[24]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


# In[25]:


movies['cast'] = movies['cast'].apply(convert3)

# In[26]:


movies.head()


# In[27]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[28]:


movies['crew'] = movies['crew'].apply(fetch_director)

# In[29]:


movies.head()

# In[30]:


movies['overview'] = movies['overview'].apply(lambda x: x.split())

# In[31]:


movies.head()

# In[32]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# In[33]:


movies.head()

# In[34]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# In[35]:


movies.head()

# In[36]:


new_df = movies[['movie_id', 'title', 'tags']]

# In[37]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# In[38]:


new_df.head()

# In[39]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# In[40]:


new_df.head()

# In[41]:


import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# In[42]:


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


# In[43]:


new_df['tags'] = new_df['tags'].apply(stem)

# In[44]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

# In[45]:


vectors = cv.fit_transform(new_df['tags']).toarray()

# In[46]:


vectors

# In[47]:


cv.get_feature_names()

# In[48]:


new_df['tags'][0]

# In[49]:


['loved', 'loving', 'love']
['love', 'love', 'love']

# In[50]:


ps.stem('dancing')

# In[51]:


stem(
    'in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')

# In[52]:


from sklearn.metrics.pairwise import cosine_similarity

# In[53]:


similarity = cosine_similarity(vectors)
similarity[0]

# In[54]:


sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]


# In[55]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[56]:


recommend('Batman Begins')

# In[57]:


new_df.iloc[1216].title

# In[58]:


import pickle

new_df.to_dict()
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

