from __future__ import absolute_import
import numpy as np

def meta_model_labels(y_true, softmax_vs):
    '''
    Generates labels for each branch.
    '''
    corrects = []
    for i,s_vs in enumerate(softmax_vs):
        y_hat = np.argmax(s_vs, axis=1)
        correct = y_hat == y_true
        corrects.append(correct)
    
    layer_labels = []
    for i in range(len(corrects)):
        # default to drop = 0
        layer_label = np.zeros(len(y_true))
        # exit = 1
        layer_label[corrects[i]] = 1
        # continue if any of the later layer can classify it correctly = 2
        layer_label[~corrects[i] & reduce(lambda x,y: x|y, corrects[i+1:],np.zeros(len(y_true),dtype=bool))] = 2
        
        layer_labels.append(layer_label)
    return np.array(layer_labels).astype(int)

    #get last layer predicitons
    #y_hat_last = np.argmax(softmax_vs[-1], axis=1)
    #last_correct = y_hat_last == y_true
    #
    #layer_labels = []
    #for s_vs in softmax_vs[:-1]:
    #    layer_label = np.zeros(len(y_true))
    #    y_hat = np.argmax(s_vs, axis=1)
    #    correct = y_hat == y_true
    #        
    #    #1 = exit
    #    layer_label[correct] = 1
    #    
    #    #2 = continue
    #    layer_label[~correct] = 2
    #    
    #    #0 = drop
    #    layer_label[~correct & ~last_correct] = 0
    #    
    #    layer_labels.append(layer_label)
    #
    #return np.array(layer_labels).astype(int)

from shadow.shadow import meta_model_labels
def test_meta_model_labels(new_labels):
    print 'unqiue',np.unique(new_labels)
    for i,new_label in enumerate(new_labels):
        print 'layer',i,'drop\t\t',0,np.sum(new_label==0)
        print 'layer',i,'exit\t\t',1,np.sum(new_label==1)
        print 'layer',i,'continue\t',2,np.sum(new_label==2)

from networks.mlp import MLP
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

def train_metamodel(train, test, num_classes, nodes=100, batchsize=256, epochs=30):
    model = L.Classifier(MLP(len(train[0][0]), nodes, num_classes))
    optimizer = chainer.optimizers.SMORMS3()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out='result')

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(epochs, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()
    
    return model