import re

import numpy as np

class UnknownSemeError(KeyError):
    pass

class SemeSet:
    def __init__(self, s):
        self.semes = s.strip().split()
        self.seme_lookup = {seme: idx for idx, seme in enumerate(self.semes)}
        self.n = len(self.semes)
        self.variables = dict()
        
    def __len__(self):
        return self.n

    def neutral(self):
        return np.zeros(self.n)

    def ones(self):
        return np.ones(self.n)
    
    def eye(self):
        return np.eye(self.n)

    def zeromat(self):
        return np.zeros((self.n, self.n))

    def register_variables(self, variables):
        self.variables.update(variables)
    
    def str2vec(self, s):
        s = s.replace('«', '').replace('»', '')
        rv = np.zeros(self.n)
        try:
            for component in s.strip().split():
                component = component.strip()
                if component[0] == '+':
                    rv[self.seme_lookup[component[1:]]] += 1
                elif component[0] == '-':
                    rv[self.seme_lookup[component[1:]]] += -1
                else:
                    rv[self.seme_lookup[component]] += 1
        except KeyError as exn:
            raise UnknownSemeError(exn.args[0])
                    
        return rv

    def vec2str(self, v):
        components = []
        for i, x in enumerate(v):
            if x >= 0.5:
                components.append('+' + self.semes[i])
            elif x <= -0.5:
                components.append('-' + self.semes[i])

        return '«' + ' '.join(components) + '»'

    def str2mat(self, s):
        rv = np.zeros((self.n, self.n))
        for component in s.strip().split():
            component = component.strip()
            try:
                a, b = component.split('>')
                if b[0] == '-':
                    rv[self.seme_lookup[a], self.seme_lookup[b[1:]]] += -1
                else:
                    rv[self.seme_lookup[a], self.seme_lookup[b]] += 1
            except ValueError:
                rv += self.variables[component]
                continue

        return rv

    def str2mats(self, *ss):
        components = []
        for s in ss:
            components.append(self.str2mat(s))
        return np.concatenate(components)
    
    def mat2str(self, m):
        components = []
        for i, v in enumerate(m):
            for j, x in enumerate(v):
                if x >= 0.5:
                    components.append(self.semes[i] + '>' + self.semes[j])
                elif x <= -0.5:
                    components.append(self.semes[i] + '>-' + self.semes[j])

        rv = ' '.join(components)
        rv = re.compile('[ ]+').sub(' ', rv)
        return rv

    def print_tensor2(self, t, msg, labels=None):
        print('-' * 20)
        print(msg)
        any_seen = False
        if labels is None:
            for v in t:
                s = self.vec2str(v)
                print(s)
                if len(s) > 2:
                    any_seen = True
        else:
            for v, label in zip(t, labels):
                s = self.vec2str(v)
                print(label, '\t', s)
                if len(s) > 2:
                    any_seen = True
        print('-' * 20)

        return any_seen
