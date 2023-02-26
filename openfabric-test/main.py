import glob
import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

import torch
from transformers import pipeline
############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass
os.chdir(r'./content')
my_files = glob.glob('*.txt')
context=""
for i in my_files:
    print(i)
    with open('./'+str(i)) as f:
        lines = f.readlines()
    s=""        
    for l in lines:
        l=l[:-1]
        s=s+" "+l
    context=context+" "+s

############################################################
# Callback function called on each execution pass
############################################################
# deepset/minilm-uncased-squad2 - 54 sec 
# squirro/albert-base-v2-squad_v2 - 120 sec 
# deepset/roberta-base-squad2 - 113 sec 
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    question_answerer = pipeline("question-answering", model='deepset/roberta-base-squad2',device=device)
    # question_answerer=question_answerer
    for text in request.text:
        # TODO Add code here
        result = question_answerer(question=text,context=context)
        response = result['answer']
        output.append(response)
    # print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
    return SimpleText(dict(text=output))
