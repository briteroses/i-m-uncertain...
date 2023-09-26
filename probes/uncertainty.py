import copy
from itertools import permutations

import numpy as np
import torch

from probes.CCS import CCS
from data.contrast import IDK_DUMMY_LABEL

class UncertaintyDetectingCCS(CCS):
    def __init__(self, x_neg, x_pos, x_idk, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x_neg = self.normalize(x_neg)
        self.x_pos = self.normalize(x_pos)
        self.x_idk = self.normalize(x_idk)
        self.d = self.x_neg.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)
    
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x_neg = torch.tensor(self.x_neg, dtype=torch.float, requires_grad=False, device=self.device)
        x_pos = torch.tensor(self.x_pos, dtype=torch.float, requires_grad=False, device=self.device)
        x_idk = torch.tensor(self.x_idk, dtype=torch.float, requires_grad=False, device=self.device)
        return x_neg, x_pos, x_idk
    
    def get_loss(self, p_neg, p_pos, p_idk):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        L_confidence = (torch.min(torch.min(p_neg, p_pos), p_idk)**2).mean(0)
        L_consistency = ((p_neg + p_pos + p_idk - 1)**2).mean(0)
        return L_confidence + L_consistency

    def get_acc(self, x_neg_test, x_pos_test, x_idk_test, y_test, includes_uncertainty = False):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x_neg = torch.tensor(self.normalize(x_neg_test), dtype=torch.float, requires_grad=False, device=self.device)
        x_pos = torch.tensor(self.normalize(x_pos_test), dtype=torch.float, requires_grad=False, device=self.device)
        x_idk = torch.tensor(self.normalize(x_idk_test), dtype=torch.float, requires_grad=False, device=self.device)
        y_test = torch.tensor(y_test, device=self.device)
        with torch.no_grad():
            p_neg, p_pos, p_idk = self.best_probe(x_neg), self.best_probe(x_pos), self.best_probe(x_idk)
        # maximize over all permutations of three p's -> three labels
        acc, coverage = -np.inf, 1
        for perm in permutations([0, 1, IDK_DUMMY_LABEL]):
            p_to_label_neg, p_to_label_pos, p_to_label_idk = perm
            predictions = torch.where(p_pos > p_neg, p_to_label_pos, p_to_label_neg)
            predictions = torch.where(p_idk > p_pos, p_to_label_idk, predictions)
            predictions = predictions.int()[:, 0]
            
            if includes_uncertainty:
                perm_acc = (predictions == y_test).float().mean()
                if acc > perm_acc:
                    acc = perm_acc
            else:
                covered = (predictions == 0) | (predictions == 1)
                perm_acc = (predictions[covered] == y_test[covered]).float().mean().item()
                if acc < perm_acc:
                    acc = perm_acc
                    coverage = covered.sum().item() / len(y_test)

        return acc, coverage
    
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x_neg, x_pos, x_idk = self.get_tensor_data()
        permutation = torch.randperm(len(x_neg))
        x_neg, x_pos, x_idk = x_neg[permutation], x_pos[permutation], x_idk[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x_neg) if self.batch_size == -1 else self.batch_size
        nbatches = len(x_neg) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            total_loss = 0
            for j in range(nbatches):
                x_neg_batch = x_neg[j*batch_size:(j+1)*batch_size]
                x_pos_batch = x_pos[j*batch_size:(j+1)*batch_size]
                x_idk_batch = x_pos[j*batch_size:(j+1)*batch_size]
            
                # probe
                p_neg, p_pos, p_idk = self.probe(x_neg_batch), self.probe(x_pos_batch), self.probe(x_idk_batch)

                # get the corresponding loss
                loss = self.get_loss(p_neg, p_pos, p_idk)
                total_loss += loss.item()

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return total_lossimport copy
from itertools import permutations

import numpy as np
import torch

from probes.CCS import CCS
from data.contrast import IDK_DUMMY_LABEL

class UncertaintyDetectingCCS(CCS):
    def __init__(self, x_neg, x_pos, x_idk, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x_neg = self.normalize(x_neg)
        self.x_pos = self.normalize(x_pos)
        self.x_idk = self.normalize(x_idk)
        self.d = self.x_neg.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)
    
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x_neg = torch.tensor(self.x_neg, dtype=torch.float, requires_grad=False, device=self.device)
        x_pos = torch.tensor(self.x_pos, dtype=torch.float, requires_grad=False, device=self.device)
        x_idk = torch.tensor(self.x_idk, dtype=torch.float, requires_grad=False, device=self.device)
        return x_neg, x_pos, x_idk
    
    def get_loss(self, p_neg, p_pos, p_idk):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        L_confidence = (torch.min(torch.min(p_neg, p_pos), p_idk)**2).mean(0)
        L_consistency = ((p_neg + p_pos + p_idk - 1)**2).mean(0)
        return L_confidence + L_consistency

    def get_acc(self, x_neg_test, x_pos_test, x_idk_test, y_test, includes_uncertainty = False):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x_neg = torch.tensor(self.normalize(x_neg_test), dtype=torch.float, requires_grad=False, device=self.device)
        x_pos = torch.tensor(self.normalize(x_pos_test), dtype=torch.float, requires_grad=False, device=self.device)
        x_idk = torch.tensor(self.normalize(x_idk_test), dtype=torch.float, requires_grad=False, device=self.device)
        y_test = torch.tensor(y_test, device=self.device)
        with torch.no_grad():
            p_neg, p_pos, p_idk = self.best_probe(x_neg), self.best_probe(x_pos), self.best_probe(x_idk)
        # maximize over all permutations of three p's -> three labels
        acc, coverage = -np.inf, 1
        for perm in permutations([0, 1, IDK_DUMMY_LABEL]):
            p_to_label_neg, p_to_label_pos, p_to_label_idk = perm
            predictions = torch.where(p_pos > p_neg, p_to_label_pos, p_to_label_neg)
            predictions = torch.where(p_idk > p_pos, p_to_label_idk, predictions)
            predictions = predictions.int()[:, 0]
            
            if includes_uncertainty:
                perm_acc = (predictions == y_test).float().mean()
                if acc > perm_acc:
                    acc = perm_acc
            else:
                covered = (predictions == 0) | (predictions == 1)
                perm_acc = (predictions[covered] == y_test[covered]).float().mean().item()
                if acc < perm_acc:
                    acc = perm_acc
                    coverage = covered.sum().item() / len(y_test)

        return acc, coverage
    
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x_neg, x_pos, x_idk = self.get_tensor_data()
        permutation = torch.randperm(len(x_neg))
        x_neg, x_pos, x_idk = x_neg[permutation], x_pos[permutation], x_idk[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x_neg) if self.batch_size == -1 else self.batch_size
        nbatches = len(x_neg) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            total_loss = 0
            for j in range(nbatches):
                x_neg_batch = x_neg[j*batch_size:(j+1)*batch_size]
                x_pos_batch = x_pos[j*batch_size:(j+1)*batch_size]
                x_idk_batch = x_pos[j*batch_size:(j+1)*batch_size]
            
                # probe
                p_neg, p_pos, p_idk = self.probe(x_neg_batch), self.probe(x_pos_batch), self.probe(x_idk_batch)

                # get the corresponding loss
                loss = self.get_loss(p_neg, p_pos, p_idk)
                total_loss += loss.item()

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return total_lossimport copy
from itertools import permutations

import numpy as np
import torch

from probes.CCS import CCS
from data.contrast import IDK_DUMMY_LABEL

class UncertaintyDetectingCCS(CCS):
    def __init__(self, x_neg, x_pos, x_idk, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x_neg = self.normalize(x_neg)
        self.x_pos = self.normalize(x_pos)
        self.x_idk = self.normalize(x_idk)
        self.d = self.x_neg.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)
    
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x_neg = torch.tensor(self.x_neg, dtype=torch.float, requires_grad=False, device=self.device)
        x_pos = torch.tensor(self.x_pos, dtype=torch.float, requires_grad=False, device=self.device)
        x_idk = torch.tensor(self.x_idk, dtype=torch.float, requires_grad=False, device=self.device)
        return x_neg, x_pos, x_idk
    
    def get_loss(self, p_neg, p_pos, p_idk):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        L_confidence = (torch.min(torch.min(p_neg, p_pos), p_idk)**2).mean(0)
        L_consistency = ((p_neg + p_pos + p_idk - 1)**2).mean(0)
        return L_confidence + L_consistency

    def get_acc(self, x_neg_test, x_pos_test, x_idk_test, y_test, includes_uncertainty = False):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x_neg = torch.tensor(self.normalize(x_neg_test), dtype=torch.float, requires_grad=False, device=self.device)
        x_pos = torch.tensor(self.normalize(x_pos_test), dtype=torch.float, requires_grad=False, device=self.device)
        x_idk = torch.tensor(self.normalize(x_idk_test), dtype=torch.float, requires_grad=False, device=self.device)
        y_test = torch.tensor(y_test, device=self.device)
        with torch.no_grad():
            p_neg, p_pos, p_idk = self.best_probe(x_neg), self.best_probe(x_pos), self.best_probe(x_idk)
        # maximize over all permutations of three p's -> three labels
        acc, coverage = -np.inf, 1
        for perm in permutations([0, 1, IDK_DUMMY_LABEL]):
            p_to_label_neg, p_to_label_pos, p_to_label_idk = perm
            predictions = torch.where(p_pos > p_neg, p_to_label_pos, p_to_label_neg)
            predictions = torch.where(p_idk > p_pos, p_to_label_idk, predictions)
            predictions = predictions.int()[:, 0]
            
            if includes_uncertainty:
                perm_acc = (predictions == y_test).float().mean()
                if acc > perm_acc:
                    acc = perm_acc
            else:
                covered = (predictions == 0) | (predictions == 1)
                perm_acc = (predictions[covered] == y_test[covered]).float().mean().item()
                if acc < perm_acc:
                    acc = perm_acc
                    coverage = covered.sum().item() / len(y_test)

        return acc, coverage
    
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x_neg, x_pos, x_idk = self.get_tensor_data()
        permutation = torch.randperm(len(x_neg))
        x_neg, x_pos, x_idk = x_neg[permutation], x_pos[permutation], x_idk[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x_neg) if self.batch_size == -1 else self.batch_size
        nbatches = len(x_neg) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            total_loss = 0
            for j in range(nbatches):
                x_neg_batch = x_neg[j*batch_size:(j+1)*batch_size]
                x_pos_batch = x_pos[j*batch_size:(j+1)*batch_size]
                x_idk_batch = x_pos[j*batch_size:(j+1)*batch_size]
            
                # probe
                p_neg, p_pos, p_idk = self.probe(x_neg_batch), self.probe(x_pos_batch), self.probe(x_idk_batch)

                # get the corresponding loss
                loss = self.get_loss(p_neg, p_pos, p_idk)
                total_loss += loss.item()

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return total_loss