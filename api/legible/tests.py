import numpy as np

from linalg import SemeSet
from layers import FeedForwardLayer, SelfAttentionLayer


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
        normaxis=1,
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
        normaxis=1,
        name='learner',
        docstring='function we are trying to change to match target',
        semes=semes)

    ema_loss = None
    ema_perc = 0.999
    
    for i in range(10000):
        seqlen = np.random.randint(3, 20)

        gradient_stores = []
        losses = []
        for b in range(64):
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
            _ = learner.backprop(dloss_dg=grad, gradient_store=gradient_store)
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
                                learning_rate=1e-8)

        
if __name__ == '__main__':
    #test_backprop_feed_forward()
    test_backprop_self_attention()
