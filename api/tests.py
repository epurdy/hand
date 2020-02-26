import numpy as np

from legible.linalg import SemeSet
from legible.layers import FeedForwardLayer, SelfAttentionLayer
from programmable_transformer import ProgrammableTransformer

def test_backprop_feed_forward():
    filter_size = 256
    embed_size = 64

    semes = SemeSet(' '.join('x' + str(i) for i in range(filter_size)))

    target = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to learn',
        semes=semes)

    learner = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to change to match target',
        semes=semes)

    ema_loss = None
    ema_perc = 0.999

    for i in range(100000):
        seqlen = np.random.randint(3, 20)
        x = np.random.randn(seqlen, embed_size)
        target_x = target.call(words=['foo'] * seqlen,
                                 vectors=x,
                                 json_log={'layers': []})
        learner_x = learner.call(words=['foo'] * seqlen,
                                 vectors=x,
                                 json_log={'layers': []})

        loss = np.linalg.norm(target_x - learner_x) ** 2
        if ema_loss is None:
            ema_loss = loss
        else:
            ema_loss = ema_perc * ema_loss + (1 - ema_perc) * loss
        print('loss', ema_loss)
        grad = 2 * (learner_x - target_x)
        gradient_store = {}
        _ = learner.backprop(dloss_dg=grad, gradient_store=gradient_store)
        learner.apply_gradients(gradient_store=gradient_store,
                                learning_rate=1e-2)


def test_backprop_self_attention():
    filter_size = 256
    embed_size = 64

    semes = SemeSet(' '.join('x' + str(i) for i in range(filter_size)))

    target = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=0,
        name='target',
        docstring='function we are trying to learn',
        semes=semes)

    learner = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=0,
        name='learner',
        docstring='function we are trying to change to match target',
        semes=semes)

    ema_loss = None
    ema_perc = 0.999

    for i in range(10000):
        seqlen = np.random.randint(30, 100)

        gradient_stores = []
        losses = []
        for b in range(16):
            x = np.random.randn(seqlen, embed_size)
            target_x, _ = target.call(words=['foo'] * seqlen,
                                   vectors=x, hidx=0, real_start=0,
                                   real_end=seqlen,
                                   json_log={'layers': []})
            learner_x, _ = learner.call(words=['foo'] * seqlen,
                                     vectors=x, hidx=0, real_start=0,
                                     real_end=seqlen,
                                     json_log={'layers': []})
            
            loss = np.linalg.norm(target_x - learner_x) ** 2
            losses.append(loss)
            grad = 2 * (learner_x - target_x)
            gradient_store = {}
            _ = learner.backprop(dloss_dg=grad,
                                 gradient_store=gradient_store)
            gradient_stores.append(gradient_store)

        loss = np.mean(losses)
        if ema_loss is None:
            ema_loss = loss
        else:
            ema_loss = ema_perc * ema_loss + (1 - ema_perc) * loss
        print('loss', ema_loss)

        avg_gradient_store = {}
        for k in gradient_stores[0]:
            avg_gradient_store[k] = np.mean([gradient_store[k]
                                             for gradient_store
                                             in gradient_stores], axis=0)

        learner.apply_gradients(gradient_store=avg_gradient_store,
                                learning_rate=1e-3)


def test_backprop_1layer():
    filter_size = 256
    embed_size = 64

    semes = SemeSet(' '.join('x' + str(i) for i in range(filter_size)))

    target_sa = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=0,
        name='target',
        docstring='function we are trying to learn',
        semes=semes)

    target_ff = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to learn',
        semes=semes)

    learner_sa = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=0,
        name='learner',
        docstring='function we are trying to change to match target',
        semes=semes)

    learner_ff = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to change to match target',
        semes=semes)
    
    ema_loss = None
    ema_perc = 0.999

    for i in range(10000):
        seqlen = np.random.randint(30, 100)

        gradient_stores = []
        losses = []
        for b in range(16):
            x = np.random.randn(seqlen, embed_size)
            target_x, _ = target_sa.call(words=['foo'] * seqlen,
                                   vectors=x, hidx=0, real_start=0,
                                   real_end=seqlen,
                                   json_log={'layers': []})
            learner_x, _ = learner_sa.call(words=['foo'] * seqlen,
                                     vectors=x, hidx=0, real_start=0,
                                     real_end=seqlen,
                                     json_log={'layers': []})

            target_x2 = target_ff.call(words=['foo'] * seqlen,
                                    vectors=target_x,
                                    json_log={'layers': []})
            learner_x2 = learner_ff.call(words=['foo'] * seqlen,
                                      vectors=learner_x,
                                      json_log={'layers': []})
            
            loss = np.linalg.norm(target_x2 - learner_x2) ** 2
            losses.append(loss)
            grad = 2 * (learner_x2 - target_x2)
            gradient_store = {}
            grad_ff = learner_ff.backprop(dloss_dg=grad,
                                          gradient_store=gradient_store)
            _ = learner_sa.backprop(dloss_dg=grad_ff,
                                    gradient_store=gradient_store)
            gradient_stores.append(gradient_store)

        loss = np.mean(losses)
        if ema_loss is None:
            ema_loss = loss
        else:
            ema_loss = ema_perc * ema_loss + (1 - ema_perc) * loss
        print('loss', ema_loss)

        avg_gradient_store = {}
        for k in gradient_stores[0]:
            avg_gradient_store[k] = np.mean([gradient_store[k]
                                             for gradient_store
                                             in gradient_stores], axis=0)

        learner_sa.apply_gradients(gradient_store=avg_gradient_store,
                                   learning_rate=1e-3)
        learner_ff.apply_gradients(gradient_store=avg_gradient_store,
                                   learning_rate=1e-3)


def test_backprop_2layer():
    filter_size = 256
    embed_size = 64

    semes = SemeSet(' '.join('x' + str(i) for i in range(filter_size)))

    target_sa1 = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=0,
        name='target',
        docstring='function we are trying to learn',
        semes=semes)

    target_ff1 = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to learn',
        semes=semes)

    target_sa2 = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=1,
        name='target',
        docstring='function we are trying to learn',
        semes=semes)

    target_ff2 = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to learn',
        semes=semes)
    
    learner_sa1 = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=0,
        name='learner',
        docstring='function we are trying to change to match target',
        semes=semes)

    learner_ff1 = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to change to match target',
        semes=semes)

    learner_sa2 = SelfAttentionLayer(
        query_mat=0.01 * np.random.randn(embed_size, embed_size),
        key_mat=0.01 * np.random.randn(embed_size, embed_size),
        value_mat=0.01 * np.random.randn(embed_size, embed_size),
        future_mask=False,
        past_mask=False,
        include_self=True,
        normaxis=1,
        name='learner',
        docstring='function we are trying to change to match target',
        semes=semes)

    learner_ff2 = FeedForwardLayer(
        mat1=0.01 * np.random.randn(embed_size, filter_size),
        bias1=0.01 * np.random.randn(filter_size),
        mat2=0.01 * np.random.randn(filter_size, embed_size),
        bias2=0.01 * np.random.randn(embed_size),
        docstring='function we are trying to change to match target',
        semes=semes)
    
    ema_loss = None
    ema_perc = 0.999

    for i in range(10000):
        seqlen = np.random.randint(30, 100)

        gradient_stores = []
        losses = []
        for b in range(16):
            x = np.random.randn(seqlen, embed_size)
            target_x, _ = target_sa1.call(words=['foo'] * seqlen,
                                   vectors=x, hidx=0, real_start=0,
                                   real_end=seqlen,
                                   json_log={'layers': []})
            learner_x, _ = learner_sa1.call(words=['foo'] * seqlen,
                                     vectors=x, hidx=0, real_start=0,
                                     real_end=seqlen,
                                     json_log={'layers': []})

            target_x2 = target_ff1.call(words=['foo'] * seqlen,
                                    vectors=target_x,
                                    json_log={'layers': []})
            learner_x2 = learner_ff1.call(words=['foo'] * seqlen,
                                      vectors=learner_x,
                                      json_log={'layers': []})

            target_x3, _ = target_sa2.call(words=['foo'] * seqlen,
                                   vectors=target_x2, hidx=0,
                                           real_start=0,
                                   real_end=seqlen,
                                   json_log={'layers': []})
            learner_x3, _ = learner_sa2.call(words=['foo'] * seqlen,
                                     vectors=learner_x2, hidx=0,
                                             real_start=0,
                                     real_end=seqlen,
                                     json_log={'layers': []})

            target_x4 = target_ff2.call(words=['foo'] * seqlen,
                                    vectors=target_x3,
                                    json_log={'layers': []})
            learner_x4 = learner_ff2.call(words=['foo'] * seqlen,
                                      vectors=learner_x3,
                                      json_log={'layers': []})
            
            loss = np.linalg.norm(target_x4 - learner_x4) ** 2
            losses.append(loss)
            grad = 2 * (learner_x4 - target_x4)
            gradient_store = {}
            grad_ff2 = learner_ff2.backprop(dloss_dg=grad,
                                          gradient_store=gradient_store)
            grad_sa2 = learner_sa2.backprop(
                dloss_dg=grad_ff2,
                gradient_store=gradient_store)
            grad_ff1 = learner_ff1.backprop(dloss_dg=grad_sa2,
                                            gradient_store=gradient_store)
            grad_sa1 = learner_sa1.backprop(dloss_dg=grad_ff1,
                                            gradient_store=gradient_store)
            
            gradient_stores.append(gradient_store)

        loss = np.mean(losses)
        if ema_loss is None:
            ema_loss = loss
        else:
            ema_loss = ema_perc * ema_loss + (1 - ema_perc) * loss
        print('loss', ema_loss)

        avg_gradient_store = {}
        for k in gradient_stores[0]:
            avg_gradient_store[k] = np.mean([gradient_store[k]
                                             for gradient_store
                                             in gradient_stores], axis=0)

        lr = 1e-3
            
        learner_sa1.apply_gradients(gradient_store=avg_gradient_store,
                                    learning_rate=lr)
        learner_ff1.apply_gradients(gradient_store=avg_gradient_store,
                                    learning_rate=lr)
        learner_sa2.apply_gradients(gradient_store=avg_gradient_store,
                                    learning_rate=lr)
        learner_ff2.apply_gradients(gradient_store=avg_gradient_store,
                                    learning_rate=lr)
        

def test_backprop_real():
    transformer = ProgrammableTransformer(program_path='syntax.att')

    transformer.fuzz(1e-6)
    
    cls_sentences = []
    with open('in_domain_train.tsv') as cola_file:
        for line in cola_file:
            source, grammatical, og_grammatical, sentence = line.split('\t')
            cls_sentences.append((sentence, bool(int(grammatical))))

    for i in range(10):
        print(i)
        np.random.shuffle(cls_sentences)
        transformer.train_batch(cls_sentences=cls_sentences[:64],
                                learning_rate=1e-4,
                                trainable=['LEXICON'])

    transformer.surface_changes()
        
        
if __name__ == '__main__':
    # test_backprop_feed_forward()
    # test_backprop_self_attention()
    # test_backprop_1layer()
    # test_backprop_2layer()
    test_backprop_real()
