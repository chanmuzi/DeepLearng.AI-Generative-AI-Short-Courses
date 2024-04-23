import os
from dotenv import load_dotenv, find_dotenv     

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage



api_key = None
dlai_endpoint = None
client = None

def load_env():
    _ = load_dotenv(find_dotenv())

def load_mistral_api_key(ret_key=False):
    load_env()
    global api_key
    global dlai_endpoint
    api_key = os.getenv("MISTRAL_API_KEY")
    dlai_endpoint = os.getenv("DLAI_MISTRAL_API_ENDPOINT")

    global client
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    
    if ret_key:
        return api_key, dlai_endpoint

def mistral(user_message, 
            model="mistral-small-latest",
            is_json=False):
    client = MistralClient(api_key=api_key, endpoint=dlai_endpoint)
    messages = [ChatMessage(role="user", content=user_message)]

    if is_json:
        chat_response = client.chat(
            model=model, 
            messages=messages,
            response_format={"type": "json_object"})
    else:
        chat_response = client.chat(
            model=model, 
            messages=messages)

    return chat_response.choices[0].message.content


def get_text_embedding(txt):
    global client
    embeddings_batch_response = client.embeddings(
        model="mistral-embed", 
        input=txt
    )
    return embeddings_batch_response.data[0].embedding

import requests
from bs4 import BeautifulSoup
import re
def get_web_article_text(url, file_save_name=None):

    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, 'html.parser')
    
    
    tag = soup.find("div", re.compile("^prose--styled"))
    text = tag.text
    
    if file_save_name:
        f = open(file_save_name, "w")
        f.write(text)
        f.close()
    return text