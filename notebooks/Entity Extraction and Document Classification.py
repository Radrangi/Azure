# Databricks notebook source
# MAGIC %md # Entity Extraction and Document Classification

# COMMAND ----------

# MAGIC %md ## 1. Setup
# MAGIC 
# MAGIC To prepare your environment, you need to install some packages and enter credentials for the Watson services.

# COMMAND ----------

# MAGIC %md ## 1.1 Install the necessary packages
# MAGIC 
# MAGIC You need the latest versions of these packages:                                                                                                                                     
# MAGIC Watson Developer Cloud: a client library for Watson services.                                                                                                                        
# MAGIC NLTK: leading platform for building Python programs to work with human language data.                                                                                                     

# COMMAND ----------

# MAGIC %md ### Install the Watson Developer Cloud package: 

# COMMAND ----------

!pip install watson-developer-cloud==1.5

# COMMAND ----------

# MAGIC %md ### Install NLTK:

# COMMAND ----------

!pip install --upgrade nltk

# COMMAND ----------

# MAGIC %md ### Install IBM Cloud Object Storage Client: 

# COMMAND ----------

!pip install ibm-cos-sdk

# COMMAND ----------

# MAGIC %md ### Now restart the kernel by choosing Kernel > Restart. 

# COMMAND ----------

# MAGIC %md ## 1.2 Import packages and libraries
# MAGIC Import the packages and libraries that you'll use:

# COMMAND ----------

import json
import watson_developer_cloud
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
  import Features, EntitiesOptions, KeywordsOptions
    
import ibm_boto3
from botocore.client import Config

import re
import nltk
import datetime
from nltk import word_tokenize,sent_tokenize,ne_chunk

import numpy as np

import unicodedata



# COMMAND ----------

# MAGIC %md ## 2. Configuration
# MAGIC Add configurable items of the notebook below

# COMMAND ----------

# MAGIC %md ### 2.1 Add your service credentials from IBM Cloud for the Watson services
# MAGIC You must create a Watson Natural Language Understanding service on IBM Cloud. Create a service for Natural Language Understanding (NLU). Insert the username and password values for your NLU in the following cell. Do not change the values of the version fields.
# MAGIC Run the cell.

# COMMAND ----------

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-03-23',
    username="",
    password="")

# COMMAND ----------

# MAGIC %md ### 2.2 Add your service credentials for Object Storage
# MAGIC You must create Object Storage service on IBM Cloud. To access data in a file in Object Storage, you need the Object Storage authentication credentials. Insert the Object Storage authentication credentials as credentials_1 in the following cell after removing the current contents in the cell.

# COMMAND ----------


# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IBM_API_KEY_ID': '',
    'IAM_SERVICE_ID': '',
    'ENDPOINT': 'https://s3.eu-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.eu-gb.bluemix.net/oidc/token',
    'BUCKET': '',
    'FILE': 'form-doc-1.txt'
}


# COMMAND ----------

# MAGIC %md ### 2.3 Global Variables
# MAGIC Add global variables.

# COMMAND ----------

sampleText='form-doc-1.txt'
ConfigFileName_Entity='config_entity_extract.txt'
ConfigFileName_Classify= 'config_legaldocs.txt'

# COMMAND ----------

# MAGIC %md ### 2.4 Configure and download required NLTK packages
# MAGIC Download the 'punkt' and 'averaged_perceptron_tagger' NLTK packages for POS tagging usage.

# COMMAND ----------

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# COMMAND ----------

# MAGIC %md ## 3. Persistence and Storage

# COMMAND ----------

# MAGIC %md ### 3.1 Configure Object Storage Client

# COMMAND ----------

cos = ibm_boto3.client('s3',
                    ibm_api_key_id=credentials_1['IBM_API_KEY_ID'],
                    ibm_service_instance_id=credentials_1['IAM_SERVICE_ID'],
                    ibm_auth_endpoint=credentials_1['IBM_AUTH_ENDPOINT'],
                    config=Config(signature_version='oauth'),
                    endpoint_url=credentials_1['ENDPOINT'])

def get_file(filename):
    '''Retrieve file from Cloud Object Storage'''
    fileobject = cos.get_object(Bucket=credentials_1['BUCKET'], Key=filename)['Body']
    return fileobject

def load_string(fileobject):
    '''Load the file contents into a Python string'''
    text = fileobject.read()
    return text

def put_file(filename, filecontents):
    '''Write file to Cloud Object Storage'''
    resp = cos.put_object(Bucket=credentials_1['BUCKET'], Key=filename, Body=filecontents)
    return resp

# COMMAND ----------

# MAGIC %md ## 4. Input Data
# MAGIC Read the data file for entity extraction from Object Store                                                                                                                               
# MAGIC Read the configuration file for augumented entity-value pairs from Object Store.

# COMMAND ----------

text_file= load_string(get_file(sampleText))
if isinstance(text_file, bytes):
    text_file = text_file.decode('utf-8') 
print(text_file)

# COMMAND ----------

config_entity = load_string(get_file(ConfigFileName_Entity)).decode('utf-8')
print(config_entity)

# COMMAND ----------

config_class = load_string(get_file(ConfigFileName_Classify)).decode('utf-8')
print(config_class)

# COMMAND ----------

# MAGIC %md ## 5. Entity Extraction
# MAGIC Extract required entities present in the document and augment the response to NLU's results

# COMMAND ----------

# MAGIC %md ### 5.1 Entites Extracted by Watson NLU

# COMMAND ----------

def analyze_using_NLU(analysistext):
    """ Call Watson Natural Language Understanding service to obtain analysis results.
    """
    response = natural_language_understanding.analyze( 
        text=analysistext,
        features=Features(keywords=KeywordsOptions()))
    response = [r['text'] for r in response['keywords']]
    return response

# COMMAND ----------

# MAGIC %md ### 5.2 Extract Entity-Value 
# MAGIC Custom entity extraction utlity fucntions for augumenting the results of Watson NLU API call

# COMMAND ----------

def POS_tagging(text):
    """ Generate Part of speech tagging of the text.
    """
    sent = re.sub(r'\n',' ',text)
    words = nltk.word_tokenize(sent)
    POSofText = nltk.tag.pos_tag(words)
    return POSofText


entval= dict()
def text_extract(reg, tag,text):
    """ Use Chunking to extract text from sentence
    """
    entities = list()
    chunkParser= nltk.RegexpParser(reg)
    chunked= chunkParser.parse(POS_tagging(text))
    #print(chunked)
    for subtree in chunked.subtrees():
        if subtree.label() == 'Chunk':
            #print(subtree.leaves())
            entities.append(subtree.leaves())
    #print(entities)
    for i in range(len(entities)):
        for j in range(len(entities[i])):
            #print(entities[i][j][0].lower())
            if tag.strip().lower() in entities[i][j][0].lower():
                #print(entities[i])
                entval.update({tag: find_NNP(entities[i],tag)})
    return entval


def find_NNP(ent, tag):
    """ Find NNP POS tags
    """
    e= ent
    for i in range(len(e)):
        if (tag not in e[i]) and (e[i][1] == 'NNP'):
            return e[i][0]



def checkValid(date):
    #f= datetime.datetime.strftime(date)
    try:
        datetime.datetime.strptime(date.strip(),"%d/%m/%Y")
        return 1
    except ValueError as err:
        print(err)
        return 0
    
def date_extract(reg, tag, text, stage_name):
    #print(reg)
    d= dict()
    dates=re.findall(tag.lower()+' '+reg,text.lower())
    print(dates)
    temp= dates[0].strip(tag.lower())
    ret= checkValid(temp)
    if ret == 1:
        d.update({tag.lower():temp})
    print(d)

def amt_extract(reg,tag,text):
    a= dict()
    amt= re.findall(reg,text)
    print(amt)
    
entities_req= list()
def entities_required(text,step, types):
    """ Extracting entities required from configuration file
    """
    configjson= json.loads(config_entity)
    for i in range(len(step)):
        if step[i]['type'] == types:
            entities_req.append(str(step[i]['tag']))
            #entities_req.append([c['tag'] for c in configjson['configuration']['class'][i]['steps'][j]])
    return entities_req

# entlist= list()
def extract_entities(config,text):
    """ Extracts entity-value pairs
    """
    configjson= json.loads(config)
    #print(configjson)
    #print(configjson['configuration']['class'][0]['steps'][0]['entity'][0]['tag'])
    classes=configjson['configuration']['class']
    #for i in range(len(classes)):
    stages= classes['stages']
    for j in range(len(stages)):
        if stages[j]['name']=='Intro':
            steps= stages[j]['steps']
            for k in range(len(steps)):
                if steps[k]['type'] == 'text':
                        #temp=entities_required(text,steps,steps[k]['type'])
                            #print(temp)
                    ent = text_extract(steps[k]['regex'],steps[k]['tag'],text)
                #elif steps[k]['type'] == 'date':
                    #dates= date_extract(steps[k]['regex1'],steps[k]['tag'],text, stages[j]['name'])
        elif stages[j]['name']=='Parties to Contract':
            steps= stages[j]['steps']
            for k in range(len(steps)):
                if steps[k]['type'] == 'text':
                        #temp=entities_required(text,steps,steps[k]['type'])
                    ent = text_extract(steps[k]['regex'],steps[k]['tag'],text)
                    #print(ent)
    
    return ent


      

# COMMAND ----------


extract_entities(config_entity, text_file)

# COMMAND ----------

# MAGIC %md ## 6. Document Classification
# MAGIC Classify documents based on entities extracted from the previous step

# COMMAND ----------

def entities_required_classification(text,config):
    """ Extracting entities from configuration file
    """
    entities_req= list()
    configjson= json.loads(config)
    for stages in configjson['configuration']['classification']['stages']:
        class_req= stages['doctype']
        entities_req.append([[c['text'],class_req] for c in stages['entities']])
    return entities_req
#entities_required_classification(text2,config1)

# COMMAND ----------

def classify_text(text, entities,config):
    """ Classify type of document from list of entities(NLU + Configuration file)
    """
    e= dict()
    entities_req= entities_required_classification(text,config)
    for i in range(len(entities_req)):
        temp= list()
        for j in range(len(entities_req[i])):
            entities_req[i][j][0]= entities_req[i][j][0].strip()
            entities_req[i][j][0]= entities_req[i][j][0].lower()
            temp.append(entities_req[i][j][0])
            res= analyze_using_NLU(text)
            #temp= temp + res
            #print text
            #text= text.decode('utf-8')
        if all(str(x) in text.lower() for x in temp) and any(str(y) in text.lower() for y in res):
            return entities_req[i][j][1]

# COMMAND ----------

def doc_classify(text,config,config1):
    """ Classify type of Document
    """
    entities= analyze_using_NLU(text)
    temp= extract_entities(config,text)
    for k,v in temp.items():
        entities.append(k)
    #print(entities)
    entities= [e.lower() for e in entities]
    entities= [e.strip() for e in entities]
    entities= set(entities)
    ret=classify_text(text,entities,config1)
    return ret

# COMMAND ----------

doc_classify(text_file,config_entity,config_class)

# COMMAND ----------

