import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_1d': lambda C, stride, affine: nn.Sequential(
        nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
    ),
    'max_pool_1d': lambda C, stride, affine: nn.Sequential(
        nn.MaxPool1d(3, stride=stride, padding=1),
    ),
    'skip_connect' : lambda C, stride, affine: Identity(),
    'conv_1x1': lambda C, stride, affine: nn.Sequential(
        nn.Conv1d(C, C, 1, stride=stride, padding=0, bias=False),
        nn.BatchNorm1d(C),
        nn.ReLU()
    ),
    'conv_3x3': lambda C, stride, affine: nn.Sequential(
        nn.Conv1d(C, C, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm1d(C),
        nn.ReLU()
    ),
    'conv_5x5': lambda C, stride, affine: nn.Sequential(
        nn.Conv1d(C, C, 5, stride=stride, padding=2, bias=False),
        nn.BatchNorm1d(C),
        nn.ReLU()
    ),
    'conv_7x7': lambda C, stride, affine: nn.Sequential(
        nn.Conv1d(C, C, 7, stride=stride, padding=3, bias=False),
        nn.BatchNorm1d(C),
        nn.ReLU()
    ),
}

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride].mul(0.)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS.keys():
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C):
        super(Cell, self).__init__()
        self.preprocess0 = nn.Conv1d(1, C, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(C)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(1+i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess0(s0)
        s0 = self.relu(s0)
        states = [s0]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)
    
class Network(nn.Module):
    def __init__(self, C, num_classes, criterion, steps=5, multiplier=1):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.cells = nn.ModuleList()
        
        cell = Cell(steps, multiplier, C)
        self.cells += [cell]

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(C * self._num_classes, 1024)
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(1024, num_classes)
        self.k = sum(1 for i in range(self._steps) for n in range(1+i))
        self._arch_parameters = nn.Parameter(1e-3 * torch.randn(self.k, len(OPS)), requires_grad=True)
        
        self._op_names = list(OPS.keys())
        self._op_index = {op: i for i, op in enumerate(self._op_names)}
        
    def forward(self, input):
        logits = self._forward(input)
        return logits

    def _forward(self, x):
        x = x.unsqueeze(1)
        for i, cell in enumerate(self.cells):
            x = cell(x, F.softmax(self._arch_parameters, dim=-1))
        out = x
        out = self.fc1(out.view(out.size(0), -1))
        out = self.relu(out)
        logits = self.regressor(out)
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)
    
    def genotype(self):
        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != self._op_index['none']))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != self._op_index['none']:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((self._op_names[k_best], j))
                    print(f"Operation from node {j} to node {i+1} is {self._op_names[k_best]}")
                start = end
                n += 1
            return gene
        gene = _parse(F.softmax(self._arch_parameters, dim=-1).data.cpu().numpy())
        return gene
    
    def arch_parameters(self):
        return [self._arch_parameters]
       
class FinalNetwork(nn.Module):
    def __init__(self, genotype, C, num_output):
        super(FinalNetwork, self).__init__()
        self._layers = nn.ModuleList()
        for name, _ in genotype:
            op = OPS[name](C, stride=1, affine=True)
            self._layers.append(op)
            
        self.preprocess0 = nn.Conv1d(1, C, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(C)
        self.fc1 = nn.Linear(C * num_output, 1024)
        self.fc2 = nn.Linear(1024, num_output)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.preprocess0(x)
        x = self.relu(x)
        for layer in self._layers:
            x = layer(x)
        out = self.fc1(x.view(x.size(0), -1))
        out = self.relu(out)
        logits = self.fc2(out)
        return logits

class Architect(object):
    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer

    def step(self, input, target):
        
        old_params = {k: v.clone() for k, v in self.network.named_parameters()}

        # Compute the gradients of the loss with respect to the architecture parameters
        self.network.zero_grad()
        loss = self.network._loss(input, target)
        loss.backward()

        # Update the architecture parameters
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if "arch_parameters" in name:
                    self.optimizer.step()

        # Restore the old network parameters
        for name, param in self.network.named_parameters():
            if name in old_params:
                param.data.copy_(old_params[name])
    