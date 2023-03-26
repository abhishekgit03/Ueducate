from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import glob,os
import json
import time
import openai
import logging
import requests
import numpy as np
import pickle
import docx2txt
import faiss
from openai.embeddings_utils import get_embedding
import matplotlib.pyplot as plt
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/status")
def status():
    return jsonify({"status":"ok"})


@app.route('/encode_docfiles', methods = ['GET','POST'])
def encode_knowledge_base():

    # encoded_data = InMemoryDocumentStore()
    file_list = []
    for file in glob.glob('docfiles/*.docx'):
        file_list.append(file)
    print("FILE LIST:",file_list)
    corpus = []
    temp=0
    para=""
    count=0
    encoded_data=[]
    
    
    for file in file_list:
        count=0
        para=""
        x = docx2txt.process(file).strip().split()
        print(x)
        for i in x:
            if(count<=90):
                para=para + i + " "
                count=count+1
                print("Para=",para)
            else:
                if(count<=150):
                    
                    if(i.endswith(".") or i.endswith("!")):
                        print("############################################Entered 1ST block")
                        para=para + i + "."
                        corpus.append(para)
                        para=""
                        count=0
                    else:
                        print("############################################Entered 2ND block")
                        para=para + i + " "
                        count=count+1
                else:
                    corpus.append(para)
                    para=""
                    count=0
        else:
            corpus.append(para)
        print(corpus)
           

    print(corpus)
    openai.api_key = json.load(open("env.json"))["OPENAI_API_KEY"]
    for i in corpus:
        embedding = get_embedding(i, engine='text-embedding-ada-002')
        encoded_data.append(embedding)
    encoded_data = np.stack(encoded_data)
    encoded_data = encoded_data.astype(np.float32)
    # encoded_data1=[]
    # try:
    #     # with open(f'embedding.pkl', 'rb+') as f:
    #     #     encoded_data1 = pickle.load(f)
    #     encoded_data1.extend(encoded_data)
    #     encoded_data1 = np.stack(encoded_data1)
    #     encoded_data1 = encoded_data1.astype(np.float32)
    #     with open(f'corpus.pkl', 'rb') as f:
    #         corpus1 = pickle.load(f)
    #         corpus1.extend(corpus)
    # except:
    #     encoded_data1=encoded_data
    #     corpus1=corpus
    # encoded_data1.append(encoded_data)
    # corpus1.append(corpus)
    with open(f'embedding.pkl', 'wb') as f:
         pickle.dump(encoded_data, f)
    with open(f'corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
    

    response = {
        'status': "sucessful",
    }

    return jsonify(response)

@app.route("/bookbot",methods = ['POST'])
def kbookbot():
    openai.api_key = json.load(open("env.json"))["OPENAI_API_KEY"]
    global datadb , userdb, knowledge_data
    req_data = request.get_json()
    user_id = req_data['user_id']
    customer_message = req_data['customer_message']
    encoded_data = pickle.load(open("embedding.pkl", "rb"))
    corpus = pickle.load(open("corpus.pkl", "rb"))
        
    d = 1536
    k = 10
    index = faiss.IndexFlatIP(d)
    index.add(encoded_data)
    print(encoded_data.shape)
    query=customer_message
    encoded_query = get_embedding(query, engine='text-embedding-ada-002')
    encoded_query = np.array([encoded_query], dtype=np.float32)
    D,I = index.search(encoded_query,k)
    paragraphs = []
    for i in range(len(I[0])):
        x = corpus[I[0][i]]
        paragraphs.append(x)

    question=customer_message
    if len(paragraphs) >= 7:
        prompt = "Imagine an AI agent that can generate human-like text based on the Customer query on the Dialogue History and Supporting Texts. The information on the Supporting Texts can be used to reinforce the AI's response. It can provide information on a wide range of topics, answer questions, and engage in conversation on a variety of subjects including finance, self-help, motivation and  personal finance. It can provide information on basic facts to more complex topics. If one has a question about a specific topic, It'll do it's best to provide a relevant and accurate response. It can also generate text in a variety of styles and formats, depending on the task at hand. It can write narratives, descriptions, articles, reports, letters, emails, poems, stories and many other types of texts.\n\n###\n\nDialogue History:\nCustomer: "+str(question)+"\n\nSupporting Texts:\nSupporting Text 1: "+str(paragraphs[0])+"\nSupporting Text 2: "+str(paragraphs[1])+"\nSupporting Text 3: "+str(paragraphs[2])+"\nSupporting Text 4: "+str(paragraphs[3])+"\nSupporting Text 5: "+str(paragraphs[4])+"\nSupporting Text 6: "+str(paragraphs[5])+"\nSupporting Text 7: "+str(paragraphs[6])+"\n\nAgent Response:"
    else:
        prompt = "Imagine an AI agent that can generate human-like text based on the Customer query on the Dialogue History and Supporting Texts. The information on the Supporting Texts can be used to reinforce the AI's response. It can provide information on a wide range of topics, answer questions, and engage in conversation on a variety of subjects including history, science, literature, art, and current events. It can provide information on basic facts to more complex topics. If one has a question about a specific topic, It'll do it's best to provide a relevant and accurate response. It can also generate text in a variety of styles and formats, depending on the task at hand. It can write narratives, descriptions, articles, reports, letters, emails, poems, stories and many other types of texts.\n\n###\n\nDialogue History:\nCustomer: "+str(question)+"\n\nSupporting Texts:\n"
        for i in range(len(paragraphs)):
            prompt += "Supporting Text "+str(i+1)+": "+str(paragraphs[i])+"\n"
        prompt += "\nAgent Response:"
    print("PROMPT:",prompt)
    final_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt = prompt,
        temperature=0.3,
        max_tokens=500,
        top_p=0.59,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["###"]
    )
        

  

    response = {
        'final_response': final_response["choices"][0]["text"],
        'status': "sucessful",
    }
    
    return jsonify(response)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 


