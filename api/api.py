from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from programmable_transformer import ProgrammableTransformer

transformer = ProgrammableTransformer(program_path='syntax.att')

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
    tokens = sentence.sentence.split()
    layers = [tokens, tokens]
    connections = [[[i] for i, t in enumerate(tokens)]]

    try:
        _, json_log = transformer.call(tokens)
    except KeyError as exn:
        return {'error': 'Unknown word: %s' % exn}
    except:
        return {'error': 'Unknown error'}

    json_log['error'] = None
    
    return json_log
