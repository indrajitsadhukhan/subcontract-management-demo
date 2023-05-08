# Step 1: Read files from directory.
# Models to try = text-search-davinci-doc-001 text-search-davinci-query-001

# pip install requests
# pip install re
# pip install bs4
# pip install collections
# pip install html
# pip install openai
# pip install tiktoken

import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import openai
import numpy as np
import tiktoken
import os
from dotenv import load_dotenv,find_dotenv
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup

from urllib.parse import urlparse
from decouple import config


# from openai.embeddings_utils import distances_from_embeddings
from openai.embeddings_utils import get_embedding, cosine_similarity



max_tokens = 100
upper_limit = 10
count=0
input_url = "https://www.acquisition.gov/far/"
current_url_domain = urlparse(input_url).netloc
encoding="utf-8"

# Azure Configurations
API_KEY = config("AZURE_OPENAI_API_KEY") 
RESOURCE_ENDPOINT = config("AZURE_OPENAI_ENDPOINT") 
DEPLOYMENT_NAME = config("COMPLETIONS_MODEL")
API_VERSION ="2022-12-01"

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 
df = pd.DataFrame()

# r = requests.get(url, headers={"api-key": API_KEY})
# print(r.text)





def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


# Create a CSV file from all the text files in text/
def text_csv():
    texts=[]
        # Create a directory to store the text files
    if not os.path.exists("processed/"):
            os.mkdir("processed/")

    # Get all the text files in the text directory
    for file in os.listdir(current_url_domain):
        # Open the file and read the text
        with open(current_url_domain+"/"+ file, "r", encoding=encoding) as f:
            text = f.read()
            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])
    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('processed/scraped.csv')
    df.head()
    openai_embeddings()


def openai_embeddings():
    # Tiktoken is used to compute the number of tokens.
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    shortened = []
    for row in df.iterrows():
        if row[1]['text'] is None:
            continue
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        else:
            shortened.append( row[1]['text'] )
        df = pd.DataFrame(shortened, columns = ['text'])
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        df.n_tokens.hist()

    # df['embeddings'] = df.text.apply(lambda x: get_embedding(x, engine='embedding')['data'][0]['embedding'])
    df['embeddings'] = df.text.apply(lambda x: get_embedding(x, engine='embedding'))
    df.to_csv('processed/embeddings_small.csv')
    df.head()


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # print("Creating context...")
    # Get the embeddings for the question
    # q_embeddings = get_embedding(question, engine='embedding')['data'][0]['embedding']
    q_embeddings = get_embedding(question, engine='embedding')
    # print("Q-embeddings")
    # Get the distances from the embeddings

    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    # print(df['embeddings'])
    # df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    df['distances']=df['embeddings'].apply(lambda x: cosine_similarity(x, q_embeddings))

    # print(df['distances'])

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])
    
    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1000,
    stop_sequence=None
):
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # print(f"Context: {context}\n")


    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    api_url = f"{RESOURCE_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/completions?api-version={API_VERSION}"
    json_data = {
         "prompt":f"Answer the question in 50 to 100 words based on the context below, and if the question can't be answered based on the context don't give any response. : ,\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            "temperature":0,
            "max_tokens":max_tokens,
    }
    headers =  {"api-key": API_KEY}

    try:
        response = requests.post(api_url, json=json_data, headers=headers)
        completion = response.json()
    
        # print the completion
        print(completion['choices'][0]['text'])
    #     # Create a completions using the question and context
    #     response = openai.Completion.create(
           
    #     )
    #     return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# Prepare embeddings calling Azure OpenAI APIs
text_csv()

# Parse Excel to extract the topics. 
while(1):
    q=input("Question: ")
    answer_question(df,question=q)
# import pandas as pd

# excel_file = "TOC.xlsx"
# xldata = pd.read_excel(excel_file,sheet_name="Sheet1")
# topics = xldata["Topics"].to_numpy()

# for topic_q in topics:
#     answer_question(df,question=topic_q)

#Generate PDF