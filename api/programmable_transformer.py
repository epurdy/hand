import numpy as np
import yaml

from legible.linalg import SemeSet
from legible.layers import (SelfAttentionLayer, FeedForwardLayer,
                            ClockLayer, ClockNormalization2,
                            MultiheadAttentionLayer,
                            WordEmbeddingLayer)

def connections_from_attention(atts, words):
    rv = [ [] for j in range(len(words)) ]
    for att in atts:
        idxs = (att > 0.3).nonzero()
        idxs = zip(*idxs)
        for i, j in idxs:
            i = int(i)
            j = int(j)
            if 2 * len(words) <= i < 3 * len(words):
                if 2 * len(words) <= j < 3 * len(words):
                    rv[j - 2 * len(words)].append(i - 2 * len(words))

    return rv


def parse_self_attention_layer(*, obj, semes, clocks):
    sa1_heads = []
    for head_name in obj:
        head = obj[head_name]
        docstring = head.pop('docstring', '')
        params = head.pop('params', dict())
        pos = head.pop('pos', dict(Q='', K=''))
        interpretant = head.pop('int')
        value_mat = semes.str2mat('')
        query_string = ' '.join([
            '%s>%s' % (seme, key)
            if not seme.startswith('-') else
            '%s>-%s' % (seme[1:], key)
            for key in head
            for seme in head[key]['Q'].split()])
        key_string = ' '.join([
            '%s>%s' % (seme, key)
            if not seme.startswith('-') else
            '%s>-%s' % (seme[1:], key)
            for key in head
            for seme in head[key]['K'].split()])
        query_mat = semes.str2mat(query_string)
        key_mat = semes.str2mat(key_string)
        value_mat = semes.str2mat(interpretant)

        query_mat += clocks.relposxform(pos['Q'], semes)
        key_mat += clocks.relposxform(pos['K'], semes)

        sa1_heads.append(
            SelfAttentionLayer(
                include_self=params.get('include_self', False),
                future_mask=params.get('future_mask', False),
                past_mask=params.get('past_mask', False),
                normaxis=params.get('normaxis', 1),
                docstring=docstring,
                query_mat=query_mat,
                key_mat=key_mat,
                value_mat=value_mat,
                name=head_name))

    return MultiheadAttentionLayer(heads=sa1_heads)

def parse_ff_layer(*, obj, semes, clocks):
    if isinstance(obj, str):
        assert not obj.strip()
        obj = dict()

    mat1  = semes.str2mat(obj.pop('mat1', ''))
    bias1 = semes.str2vec(obj.pop('bias1', ''))
    mat2  = semes.str2mat(obj.pop('mat2', ''))
    bias2 = semes.str2vec(obj.pop('bias2', ''))

    for target_key in obj:
        for join_key in obj[target_key]:
            s = obj[target_key][join_key]

            if '|' in s:
                assert ('&' not in s)
                terms = s.split('|')
                for term in terms:
                    term = term.strip()
                    if term.startswith('-'):
                        mat1 += semes.str2mat('>-'.join([term[1:],
                                                         join_key]))
                    else:
                        mat1 += semes.str2mat('>'.join([term, join_key]))
            else:
                assert ('|' not in s)
                terms = s.split('&')
                for term in terms:
                    term = term.strip()
                    if term.startswith('-'):
                        mat1 += semes.str2mat('>-'.join([term[1:],
                                                         join_key]))
                    else:
                        mat1 += semes.str2mat('>'.join([term, join_key]))
                bias1 -= (len(terms) - 1) * semes.str2vec(join_key)


            mat2 += semes.str2mat('>'.join([join_key, target_key]))



    return FeedForwardLayer(
        mat1=mat1,
        bias1=bias1,
        mat2=mat2,
        bias2=bias2)


class ProgrammableTransformer:
    def __init__(self, *, program_path):
        self.program_path = program_path

        with open(self.program_path) as file:
            self.program = yaml.load(file, Loader=yaml.FullLoader)

        self.clocks = ClockLayer(self.program.pop('CLOCKS'))

        self.og_semes = self.program.pop('SEMES')

        self.semes = SemeSet(
            '\n'.join(
                [self.og_semes] +
                self.clocks.features))

        self.og_semes = SemeSet(self.og_semes)

        variables = {k: self.semes.str2mat(v) for k,v in
                     self.program.pop('VARIABLES').items()}
        self.semes.register_variables(variables)

        self.lexicon = WordEmbeddingLayer(
            {k:self.semes.str2vec(v) for k, v in
             self.program.pop('LEXICON').items()})

        roles = self.program.pop('ROLES')
        self.roles = dict()
        for role in roles:
            self.roles[role] = self.semes.str2vec(role)
            if roles[role] is None:
                continue
            for k in roles[role]:
                if k == 'pos':
                    for signal in roles[role]['pos'].split():
                        sigvec = self.clocks.embed_signal(
                            signal=signal)
                        self.roles[role][-len(sigvec):] += sigvec
        self.roles = np.array(list(self.roles.values()))

        self.normalize = ClockNormalization2(
            semes=self.semes,
            og_semes=self.og_semes,
            clock_layer=self.clocks)

        self.self_attention = parse_self_attention_layer(
            obj=self.program.pop('SA0'), semes=self.semes,
            clocks=self.clocks)

        self.feed_forward = parse_ff_layer(
            obj=self.program.pop('FF0'), semes=self.semes,
            clocks=self.clocks)

        self.self_attention1 = parse_self_attention_layer(
            obj=self.program.pop('SA1'), semes=self.semes,
            clocks=self.clocks)

        self.feed_forward1 = parse_ff_layer(
            obj=self.program.pop('FF1'), semes=self.semes,
            clocks=self.clocks)

        self.self_attention2 = parse_self_attention_layer(
            obj=self.program.pop('SA2'), semes=self.semes,
            clocks=self.clocks)

        self.feed_forward2 = parse_ff_layer(
            obj=self.program.pop('FF2'), semes=self.semes,
            clocks=self.clocks)

        self.self_attention3 = parse_self_attention_layer(
            obj=self.program.pop('SA3'), semes=self.semes,
            clocks=self.clocks)

        self.feed_forward3 = parse_ff_layer(
            obj=self.program.pop('FF3'), semes=self.semes,
            clocks=self.clocks)

        self.self_attention4 = parse_self_attention_layer(
            obj=self.program.pop('SA4'), semes=self.semes,
            clocks=self.clocks)

        self.feed_forward4 = parse_ff_layer(
            obj=self.program.pop('FF4'), semes=self.semes,
            clocks=self.clocks)

        self.self_attention5 = parse_self_attention_layer(
            obj=self.program.pop('SA5'), semes=self.semes,
            clocks=self.clocks)

        self.feed_forward5 = parse_ff_layer(
            obj=self.program.pop('FF5'), semes=self.semes,
            clocks=self.clocks)

        assert len(self.program) == 0

    def call(self, words, verbose=False, breakpoint=None):
        json_log = {'layers': [
            {'name': 'Input',
             'heads': [],
             'tokens': words,
             'embeddings': ['' for token in words],
            }
        ]}

        output = self.lexicon.call(words=words)
        posns = self.clocks.call(n=len(words))
        neutral_posns = np.zeros_like(output)
        neutral_posns += self.semes.str2vec('+filler')
        output[:, -2 * len(self.clocks.components):] = posns
        neutral_posns[:, -2 * len(self.clocks.components):] = posns

        labels = ((['filler'] * 2 * len(words)) + words +
                  (['role'] * len(self.roles)) +
                  (['filler'] * 2 * len(words)))

        output = np.concatenate([neutral_posns, neutral_posns, output,
                                 self.roles, neutral_posns,
                                 neutral_posns], axis=0)
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Word Embedding',
             'heads': [
                 {'connections':
                  [{'in': i,
                    'out': i,
                    'info': 'Word embedding of %s' % words[i],
                  } for i in range(len(words))]}
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                            for vec in output[2 * len(words):
                                              3 * len(words)]]

            })

        for i in range(2):
            interpretant, attention = self.self_attention.call(
                output,
                semes=self.semes,
                verbose=verbose,
                breakpoint=breakpoint,
                labels=labels,
                msg='sa0')
            output += interpretant
            output = self.normalize.call(output)

            json_log['layers'].append(
                {'name': 'Self Attention 0 (iteration %d)' % i,
                 'heads': [
                     {'connections':
                      connections_from_attention(attention, words)}
                 ],
                 'tokens': words,
                 'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })

        output += self.feed_forward.call(output)
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Feed Forward 0',
             'heads': [
                 {'connections':
                  [[j] for j in range(len(words))]
                  }
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        for i in range(2):
            interpretant, attention = self.self_attention1.call(
                output,
                semes=self.semes,
                verbose=verbose,
                breakpoint=breakpoint,
                labels=labels,
                msg='sa1')
            output = output + interpretant
            output = self.normalize.call(output)

            json_log['layers'].append(
                {'name': 'Self Attention 1 (iteration %d)' % i,
                 'heads': [
                     {'connections':
                      connections_from_attention(attention, words)}
                 ],
                 'tokens': words,
                 'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
            
        output += self.feed_forward1.call(output)
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Feed Forward 1',
             'heads': [
                 {'connections':
                  [[j] for j in range(len(words))]
                  }
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        interpretant, attention = self.self_attention2.call(output,
                                            semes=self.semes,
                                            verbose=verbose,
                                                labels=labels,
                                            breakpoint=breakpoint,
                                            msg='sa2')
        output = output + interpretant
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Self Attention 2',
             'heads': [
                 {'connections':
                  connections_from_attention(attention, words)}
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        output += self.feed_forward2.call(output)
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Feed Forward 2',
             'heads': [
                 {'connections':
                  [[j] for j in range(len(words))]
                  }
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        interpretant, attention = self.self_attention3.call(output,
                                            semes=self.semes,
                                                labels=labels,
                                            verbose=verbose,
                                            breakpoint=breakpoint,
                                            msg='sa3')
        output = output + interpretant
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Self Attention 3',
             'heads': [
                 {'connections':
                  connections_from_attention(attention, words)}
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        output += self.feed_forward3.call(output)
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Feed Forward 3',
             'heads': [
                 {'connections':
                  [[j] for j in range(len(words))]
                  }
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        interpretant, attention = self.self_attention4.call(output,
                                            semes=self.semes,
                                            verbose=verbose,
                                                labels=labels,
                                            breakpoint=breakpoint,
                                            msg='sa4')
        output += interpretant
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Self Attention 4',
             'heads': [
                 {'connections':
                  connections_from_attention(attention, words)}
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        output += self.feed_forward4.call(output)
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Feed Forward 4',
             'heads': [
                 {'connections':
                  [[j] for j in range(len(words))]
                  }
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        interpretant, attention = self.self_attention5.call(output,
                                            semes=self.semes,
                                            verbose=verbose,
                                                labels=labels,
                                            breakpoint=breakpoint,
                                            msg='sa5')
        output += interpretant
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Self Attention 5',
             'heads': [
                 {'connections':
                  connections_from_attention(attention, words)}
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        output += self.feed_forward5.call(output)
        output = self.normalize.call(output)

        json_log['layers'].append(
            {'name': 'Feed Forward 5',
             'heads': [
                 {'connections':
                  [[j] for j in range(len(words))]
                  }
             ],
             'tokens': words,
             'embeddings': [self.semes.vec2str(vec)[1:-1].split()
                                for vec in output[2 * len(words):
                                                  3 * len(words)]]

                })
        
        output = output[2*len(words):3 * len(words),
                        :len(self.og_semes)]

        return output, json_log

    def parse(self, s, verbose=False, breakpoint=None):
        output, json_log = self.call(s.split(), verbose=verbose,
                                     breakpoint=breakpoint)
        print('/' + s + '/')
        self.semes.print_tensor2(output, 'output', labels=s.split())

        strings = [self.semes.vec2str(vector) for vector in output]

        return (not any('weird' in s for s in strings) and
                any('verb' in s for s in strings))

if __name__ == '__main__':
    tf = ProgrammableTransformer(program_path='syntax.att')
    tf.parse('he ate a red apple')

    lines = []
    for line in open('corpus/ling/cola_public/tokenized/in_domain_train.tsv'):
        source, grammatical, og_grammatical, sentence = line.split('\t')
        grammatical = int(grammatical)
        grammatical = (grammatical == 1)
        lines.append((sentence.strip(), grammatical))

    lines = sorted(lines)
    lines = sorted(lines, key=lambda thing: len(thing[0]))

    lines = lines[50:]

    try:
        for i, (line, grammatical) in enumerate(lines):
            print('Example #%d' % i)
            print(line)
            guess = tf.parse(line)
            assert guess == grammatical
    except KeyError:
        print('index', i)
        print('len', len(line))
        raise
    except AssertionError:
        guess = tf.parse(line, verbose=True, breakpoint='')
        print('guess', guess)
        print('grammatical', grammatical)
        print('index', i)
        print('len', len(line))
