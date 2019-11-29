import abc

import numpy as np

GAIN = 10
def sigmoid(x):
    out = x.copy()
    out = np.exp(GAIN * x) / (1 + np.exp(GAIN * x))
    out[x > 1] = 1.0
    out[x < -1] = 0.0
    return out


class EdAttentionLayer:
    def __init__(self, *, query_mat, key_mat, value_mat,
                 bonus_entities=None, include_self=True,
                 future_mask=False, past_mask=False, normaxis=-1,
                 docstring='', name=''):
        self.query_mat = query_mat
        self.key_mat = key_mat
        self.value_mat = value_mat

        self.bonus_entities = bonus_entities

        self.include_self = include_self
        self.future_mask = future_mask
        self.past_mask = past_mask
        self.normaxis = normaxis
        self.name = name
        self.docstring = docstring
        
    def call(self, *, encoder, decoder, semes=None,
             labels=None, msg='', json_log=None):
        # encoder is [key_seqlen, embedding]
        # decoder is [query_seqlen, embedding]
        assert self.bonus_entities is None
        
        # [query_seqlen, embedding]
        queries = decoder.dot(self.query_mat)

        # [key_seqlen, embedding]
        keys = encoder.dot(self.key_mat)
        values = encoder.dot(self.value_mat)
        
        # [query_seqlen, key_seqlen]
        dot_products = queries.dot(keys.T)
        attention = dot_products * 1000
        if not self.include_self:
            attention -= 100000 * np.eye(attention.shape[0])
        if self.future_mask:
            attention -= 100000 * (
                np.triu(np.ones(attention.shape[:2])) -
                np.eye(*attention.shape[:2]))
        if self.past_mask:
            attention -= 100000 * (
                np.tril(np.ones(attention.shape[:2])) -
                np.eye(*attention.shape[:2]))
        attention -= attention.max(axis=self.normaxis, keepdims=True)
        attention = np.exp(attention)
        attention /= (1e-10 + attention.sum(axis=self.normaxis,
                                            keepdims=True))
                    
        # [query_seqlen, embedding]
        interpretants = attention.dot(values)

        return interpretants, attention
        
class SelfAttentionLayer(EdAttentionLayer):
    def __init__(self, *, query_mat, key_mat, value_mat,
                 bonus_entities=None, future_mask=False, past_mask=False,
                 include_self=False, normaxis=-1, docstring='', name=''):
        super().__init__(query_mat=query_mat, key_mat=key_mat,
                         value_mat=value_mat,
                         bonus_entities=bonus_entities,
                         include_self=include_self,
                         future_mask=future_mask, past_mask=past_mask,
                         normaxis=normaxis, docstring=docstring,
                         name=name)
        
    def call(self, vectors, semes=None, labels=None, msg=''):
        return super().call(encoder=vectors, decoder=vectors,
                            semes=semes, labels=labels, msg=msg)

class MultiheadAttentionLayer:
    def __init__(self, heads):
        self.heads = heads

    def call(self, vectors,
             semes=None, labels=None, msg=''):
        attentions = []
        interpretants = []
        for head in self.heads:
            interpretant, attention = head.call(
                vectors, 
                semes=semes, labels=labels,
                msg=msg)
            interpretants.append(interpretant)
            attentions.append(attention)

        interpretants = sum(interpretants)
        
        if breakpoint == msg:
            import ipdb
            ipdb.set_trace()
            
        return interpretants, attentions

class MultiheadEdAttentionLayer:
    def __init__(self, heads):
        self.heads = heads

    def call(self, *, encoder, decoder, semes=None,
             labels=None, msg=''):
        interpretants = sum(head.call(encoder=encoder,
                                      decoder=decoder,
                                      semes=semes,
                                      labels=labels, msg=msg) for head
                            in self.heads)

        return interpretants
    
class LinearLayer:
    def __init__(self, mat, bias=None):
        self.mat = mat
        self.bias = bias
        
    def call(self, vectors):
        if self.bias is None:
            return vectors.dot(self.mat)
        else:
            return vectors.dot(self.mat) + self.bias
    
class WordEmbeddingLayer:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def call(self, words):
        return np.array([self.dictionary[word] for word in words])
 
    def call_transpose(self, vectors):
        wordlist = sorted(self.dictionary.keys())
        logits = np.array(
            [vectors.dot(self.dictionary[word].T)
             for word in wordlist]).T
        return [wordlist[logit_slice.argmax()] for logit_slice in logits]

class WordEmbeddingLayerFromFile(WordEmbeddingLayer):
    def __init__(self, *, path, semes):
        self.dictionary = dict()
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            pieces = line.split(maxsplit=1)
            word = pieces[0]
            if pieces[1:]:
                self.dictionary[word] = semes.str2vec(pieces[1])
            else:
                self.dictionary[word] = np.zeros(semes.n)
        self.dictionary['\n'] = semes.str2vec('+newline')
                
class ReluLayer:
    def call(self, vectors):
        rv = vectors.copy()
        rv[rv < 0] = 0
        return rv

class FeedForwardLayer:
    def __init__(self, mat1, bias1, mat2, bias2):
        self.mat1 = mat1
        self.mat2 = mat2
        self.bias1 = bias1
        self.bias2 = bias2

    def call(self, vectors):
        # vectors is [seqlen, embedding]
        # [seqlen, filter]
        dense = vectors.dot(self.mat1) + self.bias1

        # [seqlen, filter]
        dense_relu = ReluLayer().call(dense)

        # [seqlen, embedding]
        dense_relu_dense = dense_relu.dot(self.mat2) + self.bias2

        return dense_relu_dense
    

class Normalization2:
    def __init__(self, semes):
        self.semes = semes
        
    def call(self, vectors):
        norm_elems = []
        for elem in vectors:
            norm_elems.append(
                self.semes.str2vec(
                    self.semes.vec2str(elem)
                )
            )

        return np.array(norm_elems)

class ClockNormalization2:
    def __init__(self, semes, og_semes, clock_layer):
        self.og_semes = og_semes
        self.semes = semes
        self.clock_layer = clock_layer

    def call(self, vectors):
        vectors_og = vectors[:, :len(self.og_semes)]
        rv = Normalization2(semes=self.og_semes).call(vectors_og)
        rv = np.array([
            self.og_semes.str2vec('filler')
            if '+filler' in self.og_semes.vec2str(vector)
            else vector
            for vector in rv])
        rv = np.concatenate([rv, vectors[:, len(self.og_semes):]],
                            axis=1)
        return rv

class ClockLayer:
    def __init__(self, s):
        self.components = s.split()
        self.m = len(self.components)

        self.slow_clocks = []
        self.fast_clocks = []
        for component in self.components:
            # revolutions per utterance
            if component[0] == 'x':
                rpu = int(component[1:])
                self.slow_clocks.append(rpu)
            # words per revolution
            elif component[0] == '/':
                wpr = int(component[1:])
                self.fast_clocks.append(wpr)
            else:
                assert False, component
        self.slow_clocks = sorted(self.slow_clocks)
        self.fast_clocks = sorted(self.fast_clocks)

        self.features = []
        for rpu in self.slow_clocks:
            self.features.append('c@%drpu' % rpu)
            self.features.append('s@%drpu' % rpu)
        for wpr in self.fast_clocks:
            self.features.append('c@%dwpr' % wpr)
            self.features.append('s@%dwpr' % wpr)

    def call(self, *, n):
        # [2m x n]
        rv = []
        ts = np.arange(n, dtype=np.float32)
        for rpu in self.slow_clocks:
            theta = rpu * ts * np.pi / n
            rv.append(np.cos(theta) / np.sqrt(self.m))
            rv.append(np.sin(theta) / np.sqrt(self.m))
        for wpr in self.fast_clocks:
            theta = 2 * np.pi * ts / wpr
            rv.append(np.cos(theta) / np.sqrt(self.m))
            rv.append(np.sin(theta) / np.sqrt(self.m))

        # [n x 2m]
        return np.array(rv).T

    def relposxform(self, s, semes):
        if isinstance(s, str):
            components = s.split()
        else:
            components = [s]
        rv = semes.str2mat('')
        for component in components:
            for j, wpr in enumerate(self.fast_clocks):
                i = semes.n - 2 * (len(self.fast_clocks) - j)
                theta = 2 * np.pi * int(component) / wpr
                rv[i, i] += np.cos(theta)
                rv[i, i + 1] -= np.sin(theta)
                rv[i + 1, i] += np.sin(theta)
                rv[i + 1, i + 1] += np.cos(theta)
        return rv

    def embed_signal(self, *, signal):
        num, denom = signal.split('/')
        num, denom = int(num), int(denom)
        embedding = []
        t = num / denom
        for rpu in self.slow_clocks:
            theta = rpu * t * np.pi
            embedding.append(np.cos(theta) / np.sqrt(self.m))
            embedding.append(np.sin(theta) / np.sqrt(self.m))
        for wpr in self.fast_clocks:
            embedding.append(0)
            embedding.append(0)

        return np.array(embedding)
    
