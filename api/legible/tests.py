import numpy as np

from linalg import SemeSet
from layers import FeedForwardLayer


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
        
if __name__ == '__main__':
    test_backprop_feed_forward()
