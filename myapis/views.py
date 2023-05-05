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


from openai.embeddings_utils import distances_from_embeddings
from rest_framework.response import Response    
from rest_framework.decorators import api_view

max_tokens = 500
upper_limit = 10
count=0


"""
Functions
1. split_into_many
2. depth0
3. crawl
4. text_csv
5. openai_embeddings
6.  remove_newlines
7. create_context
8. answer_question
9. getData
10. scrape
"""


df = pd.DataFrame()
openai.api_key="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"

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


def depth0(url):
    try:
        url_text=[]
        page = urlopen(url)
        htmlcontent = (page.read()).decode("latin1")
        soup = BeautifulSoup(htmlcontent,"html.parser")
        # Loop through all the hyperlinks present in the HTML and if we get http at the begining we add them to a list
        for link in soup.find_all('a'):
            h=link.get('href')
            if h and h.startswith('http'):
                url_text.append(link.get('href'))
        return url_text,soup.get_text()
    except:
        print("Failed to do depth0 scraping.")
        return [],""

def check_domain(url):
    domain = urlparse(url).netloc
    if(domain ==""):
        return True
    if(domain != "acquisition.gov"):
        return False
    return True
BASE_URL="https://www.acquisition.gov/"
def depth1(url):
    urls,mainpage_content = depth0(url)
    print("Number of links for Depth=1: ",len(urls))
    depth1_urls=[]
    for c,link in enumerate(urls):
        if(check_domain(link)==False):
            continue
        link = BASE_URL+link
        if(c>upper_limit):
            break
        text_hyperlink_list,hyperlink_content = depth0(link)
        for link_text in text_hyperlink_list:
            depth1_urls.append(link_text)
        print("Link %d: "%(c+1),link)
        with open("text/depth1_%d.txt"%(c+1),'w',encoding="latin1",errors="ignore") as f:
            f.write(hyperlink_content)
    print("Depth1 scraping done!")
    return depth1_urls

def depth2(url):
    depth1_urls= depth1(url)
    print("Number of links for Depth=2:",len(depth1_urls))
    depth2_urls=[]
    for c,link in enumerate(depth1_urls):
        link = BASE_URL+link
        if(check_domain(link)==False):
            continue
        text_hyperlink_list,hyperlink_content = depth0(link)
        for text_link in text_hyperlink_list:
            depth2_urls.append(text_link)
        print("Link %d: "%(c+1),link)
        with open("text/depth2_%d.txt"%(c+1),'w',encoding="latin1",errors="ignore") as f:
            f.write(hyperlink_content)
    print("Depth2 scraping done!")
    return depth2_urls

def crawl(url):
    depth=1
    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")
    if depth==0:
        depth1_urls,mainpage_content = depth0(url)
        with open("text/depth_0.txt",'w',encoding="latin1",errors='ignore') as f:
            f.write(mainpage_content)
        print("Depth0 scraping done!")
    if depth==1:
       depth1(url)
    if depth==2:
        depth2(url)

    text_csv()

# Create a CSV file from all the text files in text/
def text_csv():
    texts=[]
        # Create a directory to store the text files
    if not os.path.exists("processed/"):
            os.mkdir("processed/")

    # Get all the text files in the text directory
    for file in os.listdir("text/"):
        # Open the file and read the text
        with open("text/"+ file, "r", encoding="latin-1") as f:
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

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

    df.to_csv('processed/embeddings.csv')
    df.head()



def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie



def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    print("Creating context...")
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    print("Q-embeddings")
    # Get the distances from the embeddings

    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    print(df['embeddings'])
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    print(df['distances'])

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
    max_tokens=150,
    stop_sequence=None
):
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    print("CHK-12")
    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context give the answer based on your trained data , say \"The answer is not available on the portal but according to GPT-3 the answer is: ,\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""



@api_view(['GET'])
def askQuestion(request):
    q = request.GET.get('question', '')
    print("Question: "+q)
    print("Answer Question called.")
    answer=answer_question(df,question=q)
    print("Answer:"+answer)
    return Response(answer)


@api_view(['GET'])
def scrape(request):
    url = request.GET.get('url','')
    print(url)
    crawl(url)
    check_domain("https://www.gsa.gov/about-us")
    return Response(200)
