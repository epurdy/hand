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

    _, json_log = transformer.call(tokens)
    
    return json_log
# {
#         'layers': [
#             {
#                 'name': 'Input',
#                 'heads': [],
#                 'tokens': tokens,
#                 'embeddings': ['' for token in tokens],
#             },

#             {
#                 'name': 'Word Embedding Layer',
#                 'heads': [
#                     {'connections': [[i] for i in range(len(tokens))]}
#                 ],
#                 'tokens': tokens,
#                 'embeddings': [word_embeddings[token].split()
#                                for token in tokens],
#             },

#             {
#                 'name': 'Self-Attention 1',
#                 'heads': [
#                     {'name': 'Pro1',
#                      'connections': [[i + 1] for i in range(len(tokens) -1)] + [[]]},
#                     {'name': 'Pro2',
#                      'connections': [[]] + [[i] for i in range(len(tokens) -1)]},
#                 ],
#                 'tokens': tokens,
#                 'embeddings': [word_embeddings[token].split()
#                                for token in tokens],
#             },

#             {
#                 'name': 'Feed-forward 1',
#                 'heads': [
#                     {'connections': [[i] for i in range(len(tokens))]}
#                 ],
#                 'tokens': tokens,
#                 'embeddings': [word_embeddings[token].split()
#                                for token in tokens],
#             },
            
#             {
#                 'name': 'Self-Attention 2',
#                 'heads': [
#                     {'name': 'Det1',
#                      'connections': [[i + 1] for i in range(len(tokens) -1)] + [[]]},
#                     {'name': 'Det2',
#                      'connections': [[]] + [[i] for i in range(len(tokens) -1)]},
#                 ],
#                 'tokens': tokens,
#                 'embeddings': [word_embeddings[token].split()
#                                for token in tokens],
#             },
#         ]
#         }
