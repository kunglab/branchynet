import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from scipy.stats import entropy
import types

from links.links import *
from functions import *

import time

class BranchyNet:
    def __init__(self, network, thresholdExits=None, percentTestExits=.9, percentTrainKeeps=1., lr=0.1, momentum=0.9, weight_decay=0.0001, alpha=0.001, opt="Adam", joint=True, verbose=False):
        self.opt = opt
        self.lr = lr
        self.alpha = alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.joint = joint
        self.forwardMain = None
        
        self.main = Net()
        self.models = []
        starti = 0
        curri = 0
        for link in network:
            if not isinstance(link,Branch):
                curri += 1
                self.main.add_link(link)
            else:
                net = Net(link.weight)
                net.starti = starti
                starti = curri
                net.endi = curri
                for prevlink in self.main:
                    newlink = copy.deepcopy(prevlink)
                    newlink.name = None
                    net.add_link(newlink)
                for branchlink in link:
                    newlink = copy.deepcopy(branchlink)
                    newlink.name = None
                    net.add_link(newlink)
                self.models.append(net)
        for branchlink in link:
            newlink = copy.deepcopy(branchlink)
            newlink.name = None
            self.main.add_link(newlink)
                
        if self.opt == 'MomentumSGD':
            self.optimizer = optimizers.MomentumSGD(lr=self.lr, momentum=self.momentum)
        else:
            self.optimizer = optimizers.Adam(alpha=self.alpha)
        self.optimizer.setup(self.main)

        if self.opt == 'MomentumSGD':
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(self.weight_decay))
        
        self.optimizers = []
        
        for model in self.models:
            if self.opt == 'MomentumSGD':
                optimizer = optimizers.MomentumSGD(lr=self.lr, momentum=0.9)
            else:
                optimizer = optimizers.Adam()
            optimizer.setup(model)
            
            if self.opt == 'MomentumSGD':
                optimizer.add_hook(chainer.optimizer.WeightDecay(self.weight_decay))
            
            self.optimizers.append(optimizer)            
        
        self.percentTrainKeeps = percentTrainKeeps
        self.percentTestExits = percentTestExits
        self.thresholdExits = thresholdExits
        self.clearLearnedExitsThresholds()
        
        self.verbose = verbose
        self.gpu = False
        self.xp = np
        
    def getLearnedExitsThresholds(self):
        return self.learnedExitsThresholds/self.learnedExitsThresholdsCount
        
    def clearLearnedExitsThresholds(self):
        self.learnedExitsThresholds = np.zeros(len(self.models))
        self.learnedExitsThresholdsCount = np.zeros(len(self.models))
        
    def numexits(self):
        return len(self.models)
    
    def training(self):
        for link in self.main:
            link.train = True
        for model in self.models:
            for link in model:
                link.train = True
                
    def testing(self):
        for link in self.main:
            link.train = False
        for model in self.models:
            for link in model:
                link.train = False
        
    def to_gpu(self):
        self.xp = cuda.cupy
        self.gpu = True
        self.main.to_gpu()
        [model.to_gpu() for model in self.models]        
    
    def to_cpu(self):
        self.xp = np
        self.gpu = False
        self.main.to_cpu()
        [model.to_cpu() for model in self.models]
                    
    def test(self,x,t=None):        
        numexits = []
        accuracies = []
        
        remainingXVar = x
        remainingTVar = t
            
        nummodels = len(self.models)
        numsamples = x.data.shape[0]
        
        totaltime = 0
        for i,model in enumerate(self.models):
            if remainingXVar is None or remainingTVar is None:
                numexits.append(0)
                accuracies.append(0)
                continue
            
            # if self.gpu:
            #     # Faster on GPU, less transfer
            #     smh = model.test(remainingXVar,None)
            # else:
            #     h = model.test(remainingXVar,None,model.starti,model.endi)            
            #     smh = model.test(h,None,model.endi)
            start_time = time.time()
            h = model.test(remainingXVar,model.starti,model.endi)
            endtime = time.time()
            totaltime += endtime - start_time
            
            smh = model.test(h,model.endi)
            
            softmax = F.softmax(smh)
            
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])    
            
            idx = np.zeros(entropy_value.shape[0],dtype=bool)
            if i == nummodels-1:
                idx = np.ones(entropy_value.shape[0],dtype=bool)
                numexit = sum(idx)
            else:
                if self.thresholdExits is not None:
                    #min_ent = min(entropy_value)
                    min_ent = 0
                    if isinstance(self.thresholdExits,list):
                        idx[entropy_value < min_ent+self.thresholdExits[i]] = True
                        numexit = sum(idx)
                    else:
                        idx[entropy_value < min_ent+self.thresholdExits] = True
                        numexit = sum(idx)
                else:
                    if isinstance(self.percentTestExits,list):
                        numexit = int((self.percentTestExits[i])*numsamples)
                    else:
                        numexit = int(self.percentTestExits*entropy_value.shape[0])
                    esorted = entropy_value.argsort()
                    idx[esorted[:numexit]] = True
            
            total = entropy_value.shape[0]
            numkeep = total-numexit
            numexits.append(numexit)
                        
            if self.gpu:
                xdata = h.data.get()
                # xdata = remainingXVar.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = h.data
#                 xdata = remainingXVar.data
                tdata = remainingTVar.data
            
            if numkeep > 0:
                xdata_keep = xdata[~idx]
                tdata_keep = tdata[~idx]
                remainingXVar = Variable(self.xp.array(xdata_keep,dtype=x.data.dtype),volatile=x.volatile)
                remainingTVar = Variable(self.xp.array(tdata_keep,dtype=t.data.dtype),volatile=t.volatile)
            else:
                remainingXVar = None
                remainingTVar = None
                
            if numexit > 0:
                xdata_exit = xdata[idx]
                tdata_exit = tdata[idx]                
                exitXVar = Variable(self.xp.array(xdata_exit,dtype=x.data.dtype),volatile=x.volatile)
                exitTVar = Variable(self.xp.array(tdata_exit,dtype=t.data.dtype),volatile=t.volatile)
                
                # if self.gpu:
                #     exitH = model.test(exitXVar,None)
                # else:
                #     exitH = model.test(exitXVar,None,model.endi)
                exitH = model.test(exitXVar,model.endi)
                
                accuracy = F.accuracy(exitH,exitTVar)
                
                if self.gpu:
                    accuracies.append(accuracy.data.get())
                else:
                    accuracies.append(accuracy.data)
            else:
                accuracies.append(0.)
                
        overall = 0
        for i,accuracy in enumerate(accuracies):
            overall += accuracy*numexits[i]
        overall /= np.sum(numexits)
        
        if self.verbose:
            print "numexits", numexits
            print "accuracies", accuracies
            print "overall accuracy", overall
        
        return overall, accuracies, numexits, totaltime
    
    def copy_main(self):
        self.main_copy = copy.deepcopy(self.main)
        return
    
    def run_main(self,x):
        totaltime = 0
        
        start_time = time.time()
        h = self.main.test(x)
        endtime = time.time()
        totaltime += endtime - start_time
            
        self.num_exits = [len(x.data)]
        self.runtime = totaltime
        return h.data
    
    def run(self,x):
        hs = []
        
        numexits = []
        accuracies = []
        
        remainingXVar = x
        remainingTVar = t
            
        nummodels = len(self.models)
        numsamples = x.data.shape[0]
        
        totaltime = 0
        for i,model in enumerate(self.models):
            if isinstance(remainingXVar,types.NoneType) or isinstance(remainingTVar,types.NoneType):
                break
            
            start_time = time.time()
            h = model.test(remainingXVar,model.starti,model.endi)
            endtime = time.time()
            totaltime += endtime - start_time
            
            smh = model.test(h,model.endi)            
            softmax = F.softmax(smh)
            
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])    
            
            idx = np.zeros(entropy_value.shape[0],dtype=bool)
            if i == nummodels-1:
                idx = np.ones(entropy_value.shape[0],dtype=bool)
                numexit = sum(idx)
            else:
                if self.thresholdExits is not None:
                    #min_ent = min(entropy_value)
                    min_ent = 0
                    if isinstance(self.thresholdExits,list):
                        idx[entropy_value < min_ent+self.thresholdExits[i]] = True
                        numexit = sum(idx)
                    else:
                        idx[entropy_value < min_ent+self.thresholdExits] = True
                        numexit = sum(idx)
                else:
                    if isinstance(self.percentTestExits,list):
                        numexit = int((self.percentTestExits[i])*numsamples)
                    else:
                        numexit = int(self.percentTestExits*entropy_value.shape[0])
                    esorted = entropy_value.argsort()
                    idx[esorted[:numexit]] = True
            
            total = entropy_value.shape[0]
            numkeep = total-numexit
            numexits.append(numexit)
                        
            if self.gpu:
                xdata = h.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = h.data
                tdata = remainingTVar.data
                        
            if numkeep > 0:
                xdata_keep = xdata[~idx]
                tdata_keep = tdata[~idx]
                remainingXVar = Variable(self.xp.array(xdata_keep,dtype=x.data.dtype),volatile=x.volatile)
                remainingTVar = Variable(self.xp.array(tdata_keep,dtype=t.data.dtype),volatile=t.volatile)
            else:
                remainingXVar = None
                remainingTVar = None
                
            if numexit > 0:
                xdata_exit = xdata[idx]
                tdata_exit = tdata[idx]                
                exitXVar = Variable(self.xp.array(xdata_exit,dtype=x.data.dtype),volatile=x.volatile)
                exitTVar = Variable(self.xp.array(tdata_exit,dtype=t.data.dtype),volatile=t.volatile)
                
                exitH = model.test(exitXVar,model.endi)
                hs.append(exitH.data)
        
        self.num_exits = numexits
        self.runtime = totaltime
        return np.vstack(hs)
    
    def test_model(self,model,x,t=None):
        totaltime = 0
        start_time = time.time()
        h = self.main.test(x)
        endtime = time.time()
        totaltime += endtime - start_time
            
        accuracy = F.accuracy(h,t)
        if self.gpu:
            accuracydata = accuracy.data.get()
        else:
            accuracydata = accuracy.data
        
        if self.verbose:
            print "accuracies", accuracydata
            
        return accuracydata, totaltime
    
    def train_model(self,model,x,t=None):
        self.main.zerograds()
        loss = self.main.train(x,t)
        accuracy = self.main.accuracy
        loss.backward()
        self.optimizer.update()
        if self.gpu:
            lossesdata = loss.data.get()
            accuraciesdata = accuracy.data.get()
        else:
            lossesdata = loss.data
            accuraciesdata = accuracy.data            

        if self.verbose:        
            print "losses",lossesdata
            print "accuracies",accuraciesdata
        
        return lossesdata,accuraciesdata
        
    def test_main_copy(self,x,t=None):
        return self.test_model(self.main_copy,x,t)
    
    def train_main_copy(self,x,t=None):
        return self.train_model(self.main_copy,x,t)

    def test_main(self,x,t=None):
        return self.test_model(self.main,x,t)
    
    def train_main(self,x,t=None):
        return self.train_model(self.main,x,t)

    def train_branch(self,i,x,t=None):
        return self.train_model(self.models[i],x,t)

    def get_SM(self, x):        
        numexits = []
        accuracies = []
        
          
        nummodels = len(self.models)
        numsamples = x.data.shape[0]
        exitHs = []
        h = x
        for i,model in enumerate(self.models):
            h = model.test(h, model.starti, model.endi)
            smh = model.test(h, model.endi)
            softmax = F.softmax(smh)
            exitHs.append(softmax.data)
            
                      
        return exitHs
    
    
    def find_thresholds_entropies(self, x_train, y_train, percentTrainKeeps=0.5, batchsize=1024):
        datasize = x_train.shape[0]
        nummodels = len(self.models)-1
        
        thresholds = np.zeros(nummodels)
        entropy_values = [np.array([]) for i in range(nummodels)]
        
        for i in range(0, datasize, batchsize):
            input_data = x_train[i : i + batchsize]
            label_data = y_train[i : i + batchsize]

            input_data = self.xp.asarray(input_data, dtype=self.xp.float32)
            label_data = self.xp.asarray(label_data, dtype=self.xp.int32)

            x = Variable(input_data)
            t = Variable(label_data)
            
            # FORWARD AS TEST TO GET ENTROPY AND FILTER        
            remainingXVar = x
            remainingTVar = t

            numsamples = x.data.shape[0]
            for i,model in enumerate(self.models[:-1]):
                if isinstance(remainingXVar,types.NoneType) or isinstance(remainingTVar,types.NoneType):
                    break
                loss = model.train(remainingXVar,remainingTVar)

                softmax = F.softmax(model.h)
                if self.gpu:
                    entropy_value = entropy_gpu(softmax).get()
                else:
                    entropy_value = np.array([entropy(s) for s in softmax.data])    

                entropy_values[i] = np.hstack([entropy_values[i],entropy_value])

        for i,entropy_value in enumerate(entropy_values):
            idx = np.zeros(entropy_value.shape[0],dtype=bool)
            
            total = entropy_value.shape[0]
            if isinstance(percentTrainKeeps,list):
                numkeep = (percentTrainKeeps[i])*numsamples
            else:
                numkeep = percentTrainKeeps*total
            numexit = int(total - numkeep)
            esorted = entropy_value.argsort()
            thresholds[i] = entropy_value[esorted[numexit]]
               
        return thresholds.tolist(),entropy_values
            
    def find_thresholds(self, x_train, y_train, percentTrainKeeps=0.5, batchsize=1024):  
        thresholds,_ = self.find_thresholds_entropies(x_train, y_train, percentTrainKeeps=percentTrainKeeps, batchsize=batchsize)
        return thresholds
        
    def find_entropies(self, x_train, y_train, percentTrainKeeps=0.5, batchsize=1024):  
        _,entropies = self.find_thresholds_entropies(x_train, y_train, percentTrainKeeps=percentTrainKeeps, batchsize=batchsize)
        return entropies
            
    def train(self,x,t=None):
        
        # SCATTER: copy params
        for i,link in enumerate(self.main):
            for model in self.models:
                for j,modellink in enumerate(model[:model.endi]):
                    if i==j:
                        #print i,j,link,modellink
                        modellink.copyparams(link)
        
        # RESET: zerograds
        self.main.zerograds()
        [model.zerograds() for model in self.models]
        
        
        # FORWARD
        if self.forwardMain is not None:
            mainLoss = self.main.train(x,t)
        # FORWARD AS TEST TO GET ENTROPY AND FILTER        
        remainingXVar = x
        remainingTVar = t
        
        numexits = []
        losses = []
        accuracies = []
        nummodels = len(self.models)
        numsamples = x.data.shape[0]
        for i,model in enumerate(self.models):
            if isinstance(remainingXVar,types.NoneType) or isinstance(remainingTVar,types.NoneType):
                break
            loss = model.train(remainingXVar,remainingTVar)
            losses.append(loss)
            accuracies.append(model.accuracy)
            if i == nummodels-1:
                continue
            
            softmax = F.softmax(model.h)
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])    
            
            total = entropy_value.shape[0]
            
            idx = np.zeros(total,dtype=bool)
            if i == nummodels-1:
                idx = np.ones(entropy_value.shape[0],dtype=bool)
                numexit = sum(idx)
            else:
                if self.thresholdExits is not None:
                    #min_ent = min(entropy_value)
                    min_ent = 0
                    if isinstance(self.thresholdExits,list):
                        idx[entropy_value < min_ent+self.thresholdExits[i]] = True
                        numexit = sum(idx)
                    else:
                        idx[entropy_value < min_ent+self.thresholdExits] = True
                        numexit = sum(idx)
                elif hasattr(self,'percentTrainExits') and self.percentTrainExits is not None:
                    if isinstance(self.percentTrainExits,list):
                        numexit = int((self.percentTrainExits[i])*numsamples)
                    else:
                        numexit = int(self.percentTrainExits*entropy_value.shape[0])
                    esorted = entropy_value.argsort()
                    idx[esorted[:numexit]] = True
                else:
                    if isinstance(self.percentTrainKeeps,list):
                        numkeep = (self.percentTrainKeeps[i])*numsamples
                    else:
                        numkeep = self.percentTrainKeeps*total
                    numexit = int(total - numkeep)
                    esorted = entropy_value.argsort()
                    idx[esorted[:numexit]] = True
            
            numkeep = int(total - numexit)
            numexits.append(numexit)
            
            if self.gpu:
                xdata = remainingXVar.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = remainingXVar.data
                tdata = remainingTVar.data
            
            if numkeep > 0:
                remainingXVar = Variable(self.xp.array(xdata[~idx]),volatile=x.volatile)
                remainingTVar = Variable(self.xp.array(tdata[~idx]),volatile=t.volatile)
            else:
                remainingXVar = None
                remainingTVar = None
        
        # BACKWARD
        if self.forwardMain is not None:
            mainLoss.backward()
        # BACKWARD: backward calculate gradient
        for i,loss in enumerate(losses):
            net = self.models[i]
            loss = net.weight * loss
            loss.backward()

        # GATHER: add grads to main
        if self.joint:
            # Already forwardMain 100%, no need to add grads from last
            if self.forwardMain is not None:
                models = self.models[:-1]
            else:
                models = self.models
            # added = []
            for i,link in enumerate(self.main):
                for model in models:
                    for j,modellink in enumerate(model[:model.endi]):
                        if i==j:
                            link.addgrads(modellink)
            #               if i==j and modellink.name not in added:
            #                   added.append(link.name)
            #                   link.addgrads(modellink)
        else:
            # Just the last model (main)
            for i,link in enumerate(self.main):
                for model in self.models[-1:]:
                    for j,modellink in enumerate(model[:model.endi]):
                        if i==j:
                            link.addgrads(modellink)

        # UPDATE: update parameters
        self.optimizer.update()
        [optimizer.update() for optimizer in self.optimizers]
        
        # SCATTER: copy params
        for i,link in enumerate(self.main):
            for model in self.models:
                for j,modellink in enumerate(model[:model.endi]):
                    if i==j:
                        #print i,j,link,modellink
                        modellink.copyparams(link)
                
        if self.gpu:
            lossesdata = [loss.data.get() for loss in losses]
            accuraciesdata = [accuracy.data.get() for accuracy in accuracies]
        else:
            lossesdata = [loss.data for loss in losses]
            accuraciesdata = [accuracy.data for accuracy in accuracies]
        
        if self.verbose:
            print "numexits",numexits
            print "losses",lossesdata
            print "accuracies",accuraciesdata
            
        return lossesdata,accuraciesdata
    
    def print_models(self):
        for model in self.models:
            print "----", model.starti, model.endi
            for link in model:
                print link
        print "----", self.main.starti, model.endi
        for link in self.main:
            print link
        print "----"
        