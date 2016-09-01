from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import numpy as np
import time
from itertools import product
import chainer.functions as F
from augment import augmentation

def test_suite_A(branchyNet,x_test,y_test,batchsize=10000,ps=np.linspace(0.1,1.0,10)):
    accs = []
    diffs = []
    num_exits = []
    for p in ps:
        branchyNet.percentTestExits = p
        acc,diff,num_exit,_ = test(branchyNet,x_test,y_test,batchsize=batchsize)
        accs.append(acc)
        diffs.append(diff)
        num_exits.append(num_exit)
    return ps, accs, diffs

def test_suite_B(branchyNet,x_test,y_test,batchsize=10000,ps=np.linspace(0.1,2.0,10)):
    accs = []
    diffs = []
    num_exits = []
    for p in ps:
        branchyNet.thresholdExits = p
        acc,diff,num_exit,_ = test(branchyNet,x_test,y_test,batchsize=batchsize)
        accs.append(acc)
        diffs.append(diff)
        num_exits.append(num_exit)
        
    return ps, np.array(accs), np.array(diffs)/float(len(y_test)), num_exits
    
def test_augment(branchyNet,x_test,y_test=None,batchsize=10000,main=False):
    datasize = x_test.shape[0]
    
    overall = 0.
    totaltime = 0.
    nsamples = 0
    num_exits = np.zeros(branchyNet.numexits()).astype(int)
    # finals = []
    
    sum_accuracy = 0
    num = 0
    for x,t in augmentation(x_test,y_test,batchsize):
        total = len(x)
        
        x=branchyNet.xp.asarray(x,dtype=branchyNet.xp.float32)
        t=branchyNet.xp.asarray(t,dtype=branchyNet.xp.int32)
                
        x = Variable(x, volatile=True)
        t = Variable(t, volatile=True)

        # start_time = time.time()
        if main:
            pred = branchyNet.run_main(x)
        else:
            pred = branchyNet.run(x)
        
        totaltime += branchyNet.runtime
        
        print "pred.shape",pred.shape
        pred = pred.mean(axis=0)
        acc = int(pred.argmax() == t.data[0])
        sum_accuracy += acc
        num += 1
        
        for i, exits in enumerate(branchyNet.num_exits):
            num_exits[i] += exits
    
    overall = sum_accuracy / num
    
    return overall, totaltime, num_exits

def test(branchyNet,x_test,y_test=None,batchsize=10000,main=False):
    datasize = x_test.shape[0]
    
    overall = 0.
    totaltime = 0.
    nsamples = 0
    num_exits = np.zeros(branchyNet.numexits()).astype(int)
    # finals = []
    accbreakdowns = np.zeros(branchyNet.numexits())
    
    for i in range(0, datasize, batchsize):
        input_data = x_test[i : i + batchsize]
        label_data = y_test[i : i + batchsize]

        input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
        label_data = branchyNet.xp.asarray(label_data, dtype=branchyNet.xp.int32)

        x = Variable(input_data, volatile=True)
        t = Variable(label_data, volatile=True)
        if main:
            acc, diff = branchyNet.test_main(x,t)
            #if hasattr(h.data,'get'):
            #    finals.append(h.data.get())
            #else:
            #    finals.append(h.data)
            #accuracies = [acc]
        else:
            acc, accuracies, test_exits, diff = branchyNet.test(x,t)
            for i, exits in enumerate(test_exits):
                num_exits[i] += exits
            for i in range(branchyNet.numexits()):
                accbreakdowns[i] += accuracies[i]*test_exits[i]
        
        # end_time = time.time()    
        # diff = end_time-start_time
                
        totaltime += diff
        overall += input_data.shape[0]*acc
        nsamples += input_data.shape[0]
        
    overall /= nsamples
    
    for i in range(branchyNet.numexits()):
        if num_exits[i] > 0:
            accbreakdowns[i]/=num_exits[i]
    #if len(finals) > 0:
    #    hh = Variable(np.vstack(finals),volatile=True)
    #    tt = Variable(y_test, volatile=True)
    #    overall = F.accuracy(hh,tt).data
    
    return overall, totaltime, num_exits, accbreakdowns

def get_SM(branchyNet, x_test, batchsize=10000):
    datasize = x_test.shape[0] 
    exitHs = []
    
    for i in range(0, datasize, batchsize):
        input_data = x_test[i : i + batchsize]
        input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
        x = Variable(input_data, volatile=True)
        exitHs.extend(branchyNet.get_SM(x))

    return exitHs

def traintest_augment(branchyNet,x_train,y_train,x_test,y_test,batchsize=10000,num_epoch=20,main=False):
    
    for i in range(num_epoch):
        branchyNet.training()
        plotlosses,plotaccuracies,totaltime = train_augment(branchyNet,x_train,y_train,batchsize=batchsize,num_epoch=1,main=main)
        print("train losses", plotlosses)
        print("train accuracy", plotaccuracies)
        branchyNet.testing()
        overall, totaltime, num_exits = test_augment(branchyNet,x_test,y_test,batchsize=batchsize,main=main)
        print("test accuracy", overall)

    return plotlosses,plotaccuracies,totaltime

def traintest(branchyNet,x_train,y_train,x_test,y_test,batchsize=10000,num_epoch=20,main=False,verbose=False):
    
    total_time = 0.0
    losses, accs = [], []
    test_overall, test_num_exits = [], []
    test_totaltime = 0.0
    test_accbreakdowns = []
    for i in range(num_epoch):
        branchyNet.training()
        tr_loss, tr_acc, rt = train(branchyNet,x_train,y_train,batchsize=batchsize,num_epoch=1,main=main)
        if verbose:
            print("train losses", tr_loss[0])
            print("train accuracy", tr_acc[0])
        total_time += rt
        losses.append(tr_loss[0])
        accs.append(tr_acc[0])

        branchyNet.testing()
        t_acc, t_time, t_exits, t_accbreakdowns = test(branchyNet,x_test,y_test,batchsize=batchsize,main=main)
        if verbose:
            print("test accuracy", t_accbreakdowns)
            print("test exits", t_exits)
            print("test accuracy overall", t_acc)
        test_totaltime += t_time
        test_overall.append(t_acc)
        test_num_exits.append(t_exits)
        test_accbreakdowns.append(t_accbreakdowns)

    return losses, accs, total_time, test_overall, test_totaltime, test_num_exits, test_accbreakdowns

def train_augment(branchyNet,x_train,y_train,batchsize=10000,num_epoch=20,main=False):
    datasize = x_train.shape[0]
    
    plotepochs = range(num_epoch)
    plotlosses = []
    plotaccuracies = []
    # plotnumsamples = []
    # plotexitsamples = []
    
    sum_time = 0
    
    for epoch in plotepochs:
        sum_loss = 0
        sum_accuracy = 0
        sum_time = 0
        num = 0
        for x,t in augmentation(x_train,y_train,batchsize):
            total = len(x)
            branchyNet.training()
            losses,accuracies,totaltime = train(branchyNet,x,t,batchsize=total,num_epoch=1,main=main)
            sum_loss += losses[0]*total
            sum_accuracy += accuracies[0]*total
            sum_time += totaltime
            num += total
        
        avg_loss = sum_loss / num
        avg_accuracy = sum_accuracy / num
        plotlosses.append(avg_loss)
        plotaccuracies.append(avg_accuracy)
        
    return plotlosses,plotaccuracies,sum_time

def train(branchyNet,x_train,y_train,batchsize=10000,num_epoch=20,main=False):
    datasize = x_train.shape[0]
    
    plotepochs = range(num_epoch)
    plotlosses = []
    plotaccuracies = []
    # plotnumsamples = []
    # plotexitsamples = []
    
    totaltime = 0
    
    for epoch in plotepochs:
        indexes = np.random.permutation(datasize)
        sum_loss = 0
        num = 0

        avglosses = []
        avgaccuracies = []
        # avgnumsamples = []
        # avgexitsamples = []

        for i in range(0, datasize, batchsize):
            input_data = x_train[indexes[i : i + batchsize]]
            label_data = y_train[indexes[i : i + batchsize]]

            input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
            label_data = branchyNet.xp.asarray(label_data, dtype=branchyNet.xp.int32)

            x = Variable(input_data)
            t = Variable(label_data)
            
            start_time = time.time()
            if main:
                losses,accuracies = branchyNet.train_main(x,t)
            else:
                losses,accuracies = branchyNet.train(x,t)
            end_time = time.time()    
            diff = end_time-start_time                
            totaltime += diff
            
            avglosses.append(losses)
            avgaccuracies.append(accuracies)
            # avgnumsamples.append(numsamples)
            # avgexitsamples.append(exitsamples)

        avgloss = np.mean(np.array(avglosses),0)
        avgaccuracy = np.mean(np.array(avgaccuracies),0)
        # avgnumsample = np.mean(np.array(avgnumsamples),0)
        # avgexitsample = np.mean(np.array(avgexitsamples),0)

        plotlosses.append(avgloss)
        plotaccuracies.append(avgaccuracy)
        # plotnumsamples.append(avgnumsample)
        # plotexitsamples.append(avgexitsample)
    return plotlosses,plotaccuracies,totaltime

def generate_thresholds(base_ts, num_layers):
    ts = list(product(*([base_ts]*(num_layers-1))))
    ts = [list(l) for l in ts]
      
    return ts

    
def get_inc_points(accs, diffs, ts, exits, inc_amt=-0.0005):
    idxs = np.argsort(diffs)
    accs = np.array(accs)
    diffs = np.array(diffs)
    inc_accs = [accs[idxs[0]]]
    inc_rts = [diffs[idxs[0]]]
    inc_ts = [ts[idxs[0]]]
    inc_exits = [exits[idxs[0]]]
    for i, idx in enumerate(idxs[1:]):
        #allow for small decrease
        if accs[idx] > inc_accs[-1]+inc_amt:
            inc_accs.append(accs[idx])
            inc_rts.append(diffs[idx])
            inc_ts.append(ts[idx])
            inc_exits.append(exits[idx])
    
    return inc_accs, inc_rts, inc_ts, inc_exits


def screen_branchy(branchyNet, x_test, y_test, base_ts, batchsize=1, enumerate_ts=True, verbose=False):
    '''
    Generate profile for the network compared to the baseline.
    '''
    if enumerate_ts:
        ts = generate_thresholds(base_ts, branchyNet.numexits())
    else:
        ts = base_ts
    
    ts, accs, diffs, exits = test_suite_B(branchyNet, x_test, y_test, batchsize=batchsize, ps=ts)
    
    return ts, accs, diffs, exits
    
    
def branchy_table_results(baseacc, c_basediff, g_basediff, c_accs, c_diffs, g_accs, g_diffs, inc_amt=0.01, network='Orig'):
    print 'Network,Hardware,Accuracy,Runtime,Speedup'
    fmt_str = '{},{},{:1.3f},{:1.3f},{:1.2f}'
    print fmt_str.format(network, 'cpu', baseacc, c_basediff, 1.0)
    prev_acc = 0.
    for i, (acc, diff) in enumerate(zip(c_accs, c_diffs)):
        if acc - prev_acc > inc_amt or i == len(c_accs)-1:
            prev_acc = acc
            print fmt_str.format('Branchy ' + network, 'cpu', acc - baseacc, diff, c_basediff/diff)
    
    print fmt_str.format(network, 'gpu', 0, g_basediff, 1.0)
    prev_acc = 0.
    for i, (acc, diff) in enumerate(zip(g_accs, g_diffs)):
        if acc - prev_acc > inc_amt or i == len(g_accs)-1:
            prev_acc = acc
            print fmt_str.format('Branchy ' + network, 'gpu', acc - baseacc, diff, g_basediff/diff)
            
def compute_network_times(exits, branch_times):
    total_times = []
    for exit in exits:
        total_time = 0
        for e, t in zip(exit, branch_times):
            total_time += e*t
            
        total_time /= float(np.sum(exit))
        #convert to ms
        total_time *= 1000. 
        total_times.append(total_time)

    return total_times

def compute_branch_times(net, x_test, y_test, batchsize=1, num_samples=200):
    thresholds = [10.]
    branch_times = []
    for i in xrange(len(net.models)):
        net.thresholdExits = thresholds
        _, branch_time, _ = test(net, x_test[:num_samples], y_test[:num_samples],
                                 batchsize=batchsize, main=False)
        branch_time /= float(len(y_test[:num_samples]))
        branch_times.append(branch_time)
        thresholds = [0.] + thresholds
        
    return branch_times

def get_ent_setdiff(old_ent_vals, ent_vals, percentExit=0.5):
    p_total = 0.
    diffs = []
    exited_old = np.empty(0)
    exited_new = np.empty(0)
    for i, (vo, vn) in enumerate(zip(old_ent_vals, ent_vals)):
        old_idxs = np.argsort(vo)
        new_idxs = np.argsort(vn)

        if i != 0:
            old_idxs = np.setdiff1d(old_idxs, exited_old)
            new_idxs = np.setdiff1d(new_idxs, exited_new)
                
        start_idx = 0
        end_idx = int(len(old_idxs)*percentExit)
        
        old_exits = old_idxs[start_idx:end_idx]
        new_exits = new_idxs[start_idx:end_idx]
        exited_old = np.concatenate((exited_old, old_exits))
        exited_new = np.concatenate((exited_new, new_exits))

        num_diff = len(np.setdiff1d(old_exits, new_exits))
        pct_diff = num_diff / float(end_idx-start_idx)
        diffs.append(pct_diff)
        
    return diffs
