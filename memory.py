import nengo
import nengo.spa as spa
import numpy as np

D = 32

# define the associations between words
memory = {
    'DOG': '0.7*BARK + 0.5*FUR + 0.5*TAIL + 0.2*HOUSE',
    'CAT': '0.7*MEOW + 0.5*FUR + 0.5*TAIL + 0.2*CLAW',
}

# build a transform matrix to implement the word associations
def make_memory(vocab, memory):
    t = np.zeros((vocab.dimensions, vocab.dimensions))
    for k, v in memory.items():
        t += np.outer(vocab.parse(v).v,
                      vocab.parse(k).v)
    return t
    
vocab = spa.Vocabulary(D)


model = spa.SPA()
with model:
    # the cue word
    model.cue = spa.State(D, vocab=vocab)
    
    # general working memory
    model.wm = spa.State(D, feedback=1, vocab=vocab)
    
    nengo.Connection(model.cue.output, model.wm.input,
                     transform=make_memory(vocab, memory)*0.5,
                     synapse=0.1)
                     
    # the one currently active word
    model.terms = spa.AssociativeMemory(input_vocab=vocab,
                wta_output=True, threshold=0.7)
    # use the working memory to trigger a word
    nengo.Connection(model.wm.output, model.terms.input, synapse=0.1)
    
    # have the triggered word inhibit itself in working memory
    nengo.Connection(model.terms.output, model.wm.input, synapse=0.1,
                    transform=-2)
    # have the triggered word be a bit more stable (so it doesn't
    # disappear instantly)
    nengo.Connection(model.terms.output, model.terms.input, synapse=0.1,
                    transform=0.5)
    
    


        
    