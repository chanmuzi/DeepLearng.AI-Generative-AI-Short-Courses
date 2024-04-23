import os
from dotenv import load_dotenv, find_dotenv     

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# 본인의 api key로 교체하길 권장합니다! 사용량에 제한이 있고 언제 막힐지 모르는 key입니다.
api_key = 'eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhcHAiLCJzdWIiOiIxMTE5OSIsImF1ZCI6IldFQiIsImlhdCI6MTcxMzgzNzE1NCwiZXhwIjoxNzE2NDI5MTU0fQ.0YuKJ7WM6XHgWsqEt8AS7cH3b6QQQOriDjDXl4T-jXk'
dlai_endpoint = 'http://jupyter-api-proxy.internal.dlai/rev-proxy/mistral_ai'
client = None

def load_env():
    _ = load_dotenv(find_dotenv())

def load_mistral_api_key(ret_key=False):
    load_env()
    global api_key
    global dlai_endpoint
    # api_key = os.getenv("MISTRAL_API_KEY")
    api_key = 'eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhcHAiLCJzdWIiOiIxMTE5OSIsImF1ZCI6IldFQiIsImlhdCI6MTcxMzgzNzE1NCwiZXhwIjoxNzE2NDI5MTU0fQ.0YuKJ7WM6XHgWsqEt8AS7cH3b6QQQOriDjDXl4T-jXk'
    # dlai_endpoint = os.getenv("DLAI_MISTRAL_API_ENDPOINT")
    dlai_endpoint = 'http://jupyter-api-proxy.internal.dlai/rev-proxy/mistral_ai'

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