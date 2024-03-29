import abc

import numpy as np


class EdAttentionLayer:
    def __init__(self, *, query_mat, key_mat, value_mat, include_self,
                 future_mask, past_mask, normaxis, docstring, name,
                 semes):
        self.query_mat = query_mat
        self.key_mat = key_mat
        self.value_mat = value_mat

        self.og_query_mat = query_mat.copy()
        self.og_key_mat = key_mat.copy()
        self.og_value_mat = value_mat.copy()
        
        self.include_self = include_self
        self.future_mask = future_mask
        self.past_mask = past_mask
        self.normaxis = normaxis
        self.name = name
        self.docstring = docstring
        self.semes = semes

        self.decoder = None
        self.encoder = None
        
    def fuzz(self, magnitude):
        self.query_mat += magnitude * np.random.randn(*self.query_mat.shape)
        self.key_mat += magnitude * np.random.randn(*self.query_mat.shape)
        self.value_mat += magnitude * np.random.randn(*self.query_mat.shape)

    def call(self, *, words, encoder, decoder, json_log, hidx, real_start, real_end):
        # encoder is [key_seqlen, embedding]
        self.encoder = encoder
        
        # decoder is [query_seqlen, embedding]
        self.decoder = decoder

        # [query_seqlen, embedding]
        self.queries = decoder.dot(self.query_mat)

        # [key_seqlen, embedding]
        self.keys = encoder.dot(self.key_mat)
        self.values = encoder.dot(self.value_mat)

        # [query_seqlen, key_seqlen]
        dot_products = self.queries.dot(self.keys.T)
        attention = 1000 * dot_products.copy()
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
        self.attention = (attention /
                          (1e-10 + attention.sum(axis=self.normaxis,
                                                 keepdims=True)))
        
        # [query_seqlen, embedding]
        interpretants = self.attention.dot(self.values)

        connections = []
        idxs = (self.attention >= 0.5).nonzero()
        idxs = zip(*idxs)
        for i, j in idxs:
            i = int(i)
            j = int(j)
            if real_start <= i < real_end:
                if real_start <= j < real_end:
                    info = (self.name + '\n\n' + self.docstring + 
                            '\n\nQ=%s K=%s' % (words[i - real_start],
                                               words[j - real_start]) +
                            '\n\ndelta=%s' % (self.semes.vec2str(
                                interpretants[i]))
                    )
                    connections.append(
                        {'out': i - real_start,
                         'in': j - real_start,
                         'info': info,
                         'hidx': hidx})

        return interpretants, connections

    def backprop(self, *, dloss_dg, gradient_store):
        # dloss_dg is [query_seqlen, embedding]

        # matrix multiply: A = XB -> dL/dX = (dL/dA) B.T
        # interpretants = attention.dot(values)
        # [query_seqlen, key_seqlen]
        dloss_datt = dloss_dg.dot(self.values.T)

        # matrix multiply: A = XB -> dL/dB = X.T (dL/dA)
        # interpretants = attention.dot(values)
        # [key_seqlen, embedding]
        dloss_dvalues = self.attention.T.dot(dloss_dg)

        # matrix multiply: A = XB -> dL/dB = X.T (dL/dA)
        # values = encoder.dot(self.value_mat)
        # [embedding, embedding]
        # TODO double check that this isn't transposed
        gradient_store[id(self), 'value_mat'] = self.encoder.T.dot(
            dloss_dvalues)
        
        # [query_seqlen, key_seqlen]
        if self.normaxis == 0:
            dloss_ddp_cols = []
            for t in range(self.attention.shape[1]):
                att_t = self.attention[:, t:t+1].copy().T
                if not self.include_self:
                    att_t[0, t] = 0
                dloss_datt_t = dloss_datt[:, t:t+1].T
                dloss_ddp_t = dloss_datt_t.dot(np.diag(att_t[0]) - att_t.T.dot(att_t))
                dloss_ddp_cols.append(dloss_ddp_t[0])
            dloss_ddp = 1000 * np.array(dloss_ddp_cols).T
        elif self.normaxis == 1:
            dloss_ddp_rows = []
            for t in range(len(self.attention)):
                att_t = self.attention[t:t+1, :].copy()
                if not self.include_self:
                    att_t[0, t] = 0
                dloss_datt_t = dloss_datt[t:t+1, :]
                #dloss_ddp_t = dloss_datt_t.dot( - att_t.T.dot(att_t))
                dloss_ddp_t = dloss_datt_t.dot(np.diag(att_t[0]) - att_t.T.dot(att_t))
                dloss_ddp_rows.append(dloss_ddp_t[0])
            dloss_ddp = 1000 * np.array(dloss_ddp_rows)

        # dloss_ddp = dloss_datt
        
        # matrix multiply: A = XB -> dL/dX = (dL/dA) B.T
        # dot_products = self.queries.dot(self.keys.T)
        # [query_seqlen, embedding]
        dloss_dqueries = dloss_ddp.dot(self.keys)

        # matrix multiply: A = XB -> dL/dB = X.T (dL/dA)
        # dot_products = self.queries.dot(self.keys.T)
        # [key_seqlen, embedding]
        dloss_dkeys = self.queries.T.dot(dloss_ddp).T
        #dloss_dkeys = dloss_ddp.T.dot(self.queries)

        # matrix multiply: A = XB -> dL/dB = X.T (dL/dA)
        # self.queries = decoder.dot(self.query_mat)
        # [embedding, embedding]
        gradient_store[id(self), 'query_mat'] = self.decoder.T.dot(
            dloss_dqueries)

        gradient_store[id(self), 'key_mat'] = self.encoder.T.dot(
            dloss_dkeys)
        
        # matrix multiply: A = XB -> dL/dX = (dL/dA) B.T
        # values = encoder.dot(self.value_mat)
        # keys = encoder.dot(self.key_mat)
        # [key_seqlen, embedding]
        dloss_dencoder = (dloss_dvalues.dot(self.value_mat.T) +
                          dloss_dkeys.dot(self.key_mat.T))

        # matrix multiply: A = XB -> dL/dX = (dL/dA) B.T
        # queries = decoder.dot(self.query_mat)
        dloss_ddecoder = dloss_dqueries.dot(self.query_mat.T)

        return dloss_dencoder, dloss_ddecoder
        
    def apply_gradients(self, *, gradient_store, learning_rate):
        self.query_mat -= learning_rate * gradient_store[id(self),
                                                         'query_mat']
        self.key_mat -= learning_rate * gradient_store[id(self),
                                                       'key_mat']
        self.value_mat -= learning_rate * gradient_store[id(self),
                                                         'value_mat']

    

class SelfAttentionLayer(EdAttentionLayer):
    def __init__(self, *, query_mat, key_mat, value_mat, future_mask,
                 past_mask, include_self, normaxis, docstring, name,
                 semes):
        super().__init__(query_mat=query_mat, key_mat=key_mat,
                         value_mat=value_mat,
                         include_self=include_self,
                         future_mask=future_mask, past_mask=past_mask,
                         normaxis=normaxis, docstring=docstring,
                         name=name, semes=semes)

    def call(self, *, words, vectors, json_log, hidx, real_start,
             real_end):
        return super().call(words=words, encoder=vectors,
                            decoder=vectors, json_log=json_log,
                            hidx=hidx, real_start=real_start,
                            real_end=real_end)

    def backprop(self, *, dloss_dg, gradient_store):
        denc, ddec = super().backprop(dloss_dg=dloss_dg,
                                      gradient_store=gradient_store)
        return denc + ddec
    
class MultiheadAttentionLayer:
    def __init__(self, *, heads, semes):
        self.heads = heads
        self.semes = semes

    def fuzz(self, magnitude):
        for head in self.heads:
            head.fuzz(magnitude)
        
    @classmethod
    def parse_layer(cls, *, obj, semes, clocks):
        pass

    def backprop(self, *, dloss_dg, gradient_store):
        # [seqlen, embedding]
        return dloss_dg + sum(head.backprop(dloss_dg=dloss_dg,
                                            gradient_store=gradient_store)
                              for head in self.heads)

    def apply_gradients(self, *, gradient_store, learning_rate):
        for head in self.heads:
            head.apply_gradients(gradient_store=gradient_store,
                                 learning_rate=learning_rate)
    
    def call(self, *, words, vectors, json_log, real_start, real_end):
        connections = []
        interpretants = []
        for hidx, head in enumerate(self.heads):
            interpretant, new_connections = head.call(
                words=words,
                vectors=vectors, json_log=json_log, hidx=hidx + 1,
                real_start=real_start, real_end=real_end)
            interpretants.append(interpretant)
            connections.extend(new_connections)

        output = vectors + sum(interpretants)

        connections.extend([
            {'out': i,
             'in': i,
             'info': 'Residual connection',
             'hidx': 0,
             }
            for i in range(len(words))
            ])
        
        json_log['layers'].append(
            {'name': 'Self Attention',
             'desc': 'Self-attention layer',
             'heads': ['Residual'] + [
                 head.name for head in self.heads
             ],
             'head_descs': ['Residual connection'] + [
                 head.docstring for head in self.heads
             ],
             'connections': connections,
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                            for vec in output[2 * len(words):
                                              3 * len(words)]]

            })


        return output


class MultiheadEdAttentionLayer:
    def __init__(self, heads):
        self.heads = heads

    def call(self, *, encoder, decoder, json_log):
        interpretants = sum(head.call(encoder=encoder,
                                      decoder=decoder,
                                      json_log=json_log) for
                            head in self.heads)

        return interpretants


class WordEmbeddingLayer:
    def __init__(self, *, semes, dictionary):
        self.semes = semes
        self.dictionary = dictionary
        self.unknown = self.semes.str2vec('+unknownword')

        self.words = None
        
    def call(self, words, json_log):
        self.words = words
        
        rv = np.array([self.dictionary.get(word, self.unknown)
                       for word in words])

        json_log['layers'].append(
            {'name': 'Word Embedding',
             'desc': 'Each word is mapped to a vector representation',
             'connections': [
                 {'in': i,
                  'out': i,
                  'info': """Word embedding of "%s" is:
%s""" % (word, self.semes.vec2str(rv[i])),
                  'hidx': 0,
                  }
                  for i, word in enumerate(words)
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                            for vec in rv],
            })

        return rv

    def fuzz(self, magnitude):
        raise NotImplementedError
    
    def backprop(self, *, dloss_dg, gradient_store):
        gradient_store[id(self), 'embeddings'] = dloss_dg.copy()

    def apply_gradients(self, *, gradient_store, learning_rate):
        for word, dword in zip(self.words,
                               gradient_store.pop((id(self), 'embeddings'))):
            if word not in self.dictionary:
                self.dictionary[word] = self.semes.neutral()
            self.dictionary[word] -= learning_rate * dword
        
    def call_transpose(self, vectors, json_log):
        wordlist = sorted(self.dictionary.keys())
        logits = np.array(
            [vectors.dot(self.dictionary[word].T)
             for word in wordlist]).T
        return [wordlist[logit_slice.argmax()] for logit_slice in logits]

class ClassificationLayer:
    def __init__(self, *, semes):
        self.semes = semes

        self.grammatical = None
        self.input = None
        
    def call(self, *, words, vectors, json_log):
        self.input = vectors

        # max pool -> [embedding]
        self.maxpool = np.max(vectors, axis=0)

        self.grammatical = ('+weird' not in self.semes.vec2str(
            self.maxpool))
        if self.grammatical:
            token = 'GRAMMATICAL'
        else:
            token = 'UNGRAMMATICAL'

        desc = """We max-pool over the sequence dimension and then 
        project down to +weird"""
            
        json_log['layers'].append(
            {'name': 'Max Pool + Classify',
             'desc': desc,
             'connections': [
                 {'in': i,
                  'out': 0,
                  'info': desc,
                  'hidx': 0,
                  }
                  for i, word in enumerate(words)
             ],
             'tokens': [token],
             'embeddings': [self.semes.vec2str(
                 self.maxpool)[1:-1].split()]
            })

        return self.grammatical

    def backprop(self, *, gradient_store):
        assert self.grammatical is not None

        if self.grammatical:
            # grammatical -> we assume that we want to flip it to
            # ungrammatical, so higher grammaticality means higher
            # loss
            dloss_dgram = 1
        else:
            # ungrammatical -> we assume that we want to flip it to
            # grammatical, so higher grammaticality means lower loss
            dloss_dgram = -1

        # []
        dloss_dweird = -dloss_dgram

        # [embedding]
        dloss_dmaxpool = dloss_dweird * self.semes.str2vec('+weird')

        thing1 = (self.input == self.maxpool.reshape(1, -1))

        # [seqlen, embedding]
        dloss_dinput = dloss_dmaxpool * thing1
            

        return dloss_dinput
        
class ReluLayer:
    def call(self, *, vectors):
        rv = vectors.copy()
        rv[rv < 0] = 0
        return rv


class FeedForwardLayer:
    def __init__(self, *, mat1, bias1, mat2, bias2, docstring, semes):
        self.mat1 = mat1
        self.mat2 = mat2
        self.bias1 = bias1
        self.bias2 = bias2
        self.docstring = docstring
        self.semes = semes

        self.og_mat1 = mat1.copy()
        self.og_mat2 = mat2.copy()
        self.og_bias1 = bias1.copy()
        self.og_bias2 = bias2.copy()
        
        # tensors used during fwd and bkwd passes
        self.input = None
        self.dense = None
        self.dense_relu = None
        self.dense_relu_dense = None
        self.output = None


    def fuzz(self, magnitude):
        self.mat1 += magnitude * np.random.randn(*self.mat1.shape)
        self.mat2 += magnitude * np.random.randn(*self.mat2.shape)
        self.bias1 += magnitude * np.random.randn(*self.bias1.shape)
        self.bias2 += magnitude * np.random.randn(*self.bias2.shape)
        
    def call(self, *, words, vectors, json_log):
        # [seqlen, embedding]
        self.input = vectors
        
        # vectors is [seqlen, embedding]
        # [seqlen, filter]
        self.dense = vectors.dot(self.mat1) + self.bias1

        # [seqlen, filter]
        self.dense_relu = ReluLayer().call(vectors=self.dense)

        # [seqlen, embedding]
        self.dense_relu_dense = self.dense_relu.dot(
            self.mat2) + self.bias2

        self.output = vectors + self.dense_relu_dense

        json_log['layers'].append(
            {'name': 'Feed Forward',
             'desc': self.docstring,
             'heads': [],
             'connections': [
                 {'in': i,
                  'out': i,
                  'info': self.docstring,
                  'hidx': 0}
                 for i in range(len(words))
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                            for vec in self.output[2 * len(words):
                                                   3 * len(words)]]

            })

        return self.output

    def backprop(self, *, dloss_dg, gradient_store):
        # loss = f(g(h(x))), where f is later layers, g is this
        # layer, and h is previous layers
        #
        # we want the gradients of this layer's weights.
        #
        # we also need to produce dloss/dh to pass to previous layer

        # dloss_dg is [seqlen, embedding], because it's the derivative
        # of a scalar (loss) by a [seqlen, embedding] tensor (g)
        
        # broadcasting add -> sum dloss_dg over broadcast axis
        # [embedding]
        gradient_store[id(self), 'bias2'] = dloss_dg.sum(axis=0)

        # matrix multiply: A = XB -> dL/dB = X.T (dL/dA)
        # [filter, embedding]
        gradient_store[id(self), 'mat2'] = self.dense_relu.T.dot(dloss_dg)

        # matrix multiply: A = XB -> dL/dX = (dL/dA) B.T
        # [seqlen, filter]
        dloss_ddenserelu = dloss_dg.dot(self.mat2.T)

        # relu: zero out coordinates that relu cut off
        # [seqlen, filter]
        dloss_ddense = dloss_ddenserelu.copy()
        dloss_ddense[self.dense < 0] = 0
        # this makes intuitive sense to me - at zero, don't claim you
        # can decrease the value, but do claim you can increase it
        dloss_ddense[np.logical_and(self.dense == 0,
                                    dloss_ddenserelu < 0)] = 0
        
        # broadcasting add -> sum dloss/ddense over broadcast axis
        # [filter]
        gradient_store[id(self), 'bias1'] = dloss_ddense.sum(axis=0)

        # matrix multiply: A = XB -> dL/dB = X.T (dL/dA)
        # [embedding, filter]
        gradient_store[id(self), 'mat1'] = self.input.T.dot(dloss_ddense)

        # matrix multiply: A = XB -> dL/dX = (dL/dA) B.T
        # [seqlen, embedding]
        dloss_dinput = dloss_ddense.dot(self.mat1.T)

        # finally, account for residual connection
        dloss_dinput = dloss_dg + dloss_dinput

        return dloss_dinput

    def apply_gradients(self, *, gradient_store, learning_rate):
        self.mat1 -= learning_rate * gradient_store[id(self), 'mat1']
        self.bias1 -= learning_rate * gradient_store[id(self), 'bias1']
        self.mat2 -= learning_rate * gradient_store[id(self), 'mat2']
        self.bias2 -= learning_rate * gradient_store[id(self), 'bias2']
    
class Normalization2:
    def __init__(self, semes):
        self.semes = semes

    def call(self, vectors, json_log):
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

    def call(self, vectors, json_log):
        vectors_og = vectors[:, :len(self.og_semes)]
        rv = Normalization2(semes=self.og_semes).call(vectors_og,
                                                      json_log=json_log)
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

    def call(self, *, n, json_log):
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
