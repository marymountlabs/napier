from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import re
import spacy
from spacy import displacy
from whoosh.index import create_in, open_dir
from whoosh.fields import *
import requests
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Function for data preprocessing
def preprocess_data(data: str) -> str:
    data = data.lower()
    data = re.sub(r"[^a-zA-Z0-9]", " ", data)
    return data

# Function for extracting structured data
def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

# Function for linking entities
def link_entities(entities):
    nlp = spacy.load("en_core_web_sm")
    linked_entities = []
    for ent in entities:
        doc = nlp(ent[0])
        if doc.ents:
            linked_entities.append((doc.ents[0].text,doc.ents[0].label_,doc.ents[0].kb_id_))
    return linked_entities

def index_data(data: list[str]):
    schema = Schema(content=TEXT(stored=True))
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    for d in data:
        writer.add_document(content=d)
    writer.commit()
    return ix

def fine_tune_gpt3(prompt, context):
    openai.api_key = os.getenv('API_Key')

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        temperature=0,
    )

    return response["choices"][0]["text"]


def search_index(ix, query):
    with ix.searcher() as searcher:
        query
