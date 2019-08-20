import pandas as pd
import random
from app.chatbot.chatbot import chat_run
import nltk
import pickle
#import time

#start_time = time.time()

df = pd.read_csv('Q&A.tsv', delimiter='\t', encoding='utf-8')

questions = [q.lower() for q in df['Questions']]
responses = [r.lower() for r in df['Responses']]

def generate_unique_response(string):
    edit_distances = []
    for index, question in enumerate(questions, start=0):
        edit_distance_str = nltk.edit_distance(string, question)
        if edit_distance_str >= 0 and edit_distance_str <= 2:
            edit_distances.append((edit_distance_str, index))
    if len(edit_distances) > 0:
        response = random.choice(edit_distances)
        #print(edit_distances)
        #print(response)
        #print(response[1])
        #print(responses[response[1]])
        #print("----------------------")
        return responses[response[1]]
    else:
        #print("CHATBOT")
        return chat_run(string)

#end_time = time.time()

#print("Total time elapsed:", end_time-start_time)