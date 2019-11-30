import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from programmable_transformer import ProgrammableTransformer

transformer = ProgrammableTransformer(program_path='syntax.att')

sentences = []
with open('in_domain_train.tsv') as cola_file:
    for line in cola_file:
        source, grammatical, og_grammatical, sentence = line.split('\t')
        sentences.append(sentence)

app = FastAPI()

origins = [
        "http://localhost:3000",
    ]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class Sentence(BaseModel):
    sentence: str
    random: bool

word_embeddings = dict(
    he="+pro +noun +masc +sg +3rd",
    ate="+eat +verb +preterite",
    a="+determiner +sg",
    red="+red +adjective",
    apple="+apple +noun",
)
word_embeddings['.'] = ''
    
@app.post("/parse")
def parse_sentence(sentence: Sentence):
    if sentence.random:
        sentence = np.random.choice(sentences)
    else:
        sentence = sentence.sentence

    tokens = sentence.lower().split()

    try:
        cls, json_log = transformer.call(tokens)
        json_log['cls'] = cls
    except KeyError as exn:
        return {'error': '404 Unknown word: %s' % exn}
    except Exception as exn:
        print(exn)
        return {'error': '500 Unknown server error'}

    json_log['error'] = None

    with open('syntax.att') as program:
        json_log['program'] = program.read()
    
    return json_log
