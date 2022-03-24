from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn

class TreeLoss(nn.Module):
    def __init__(self, hierarchy, total_nodes, levels, device):
        super(TreeLoss, self).__init__()
        self.stateSpace = self.generateStateSpace(hierarchy, total_nodes, levels).to(device)

    def forward(self, fs, labels, device):
        index = torch.mm(self.stateSpace, fs.T)
        joint = torch.exp(index)
        z = torch.sum(joint, dim=0)
        loss = torch.zeros(fs.shape[0], dtype=torch.float64).to(device)
        for i in range(len(labels)):
            marginal = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.stateSpace[:, labels[i]] > 0)[0]))
            loss[i] = -torch.log(marginal / z[i])
        return torch.mean(loss)

    def inference(self, fs, device):
        with torch.no_grad():
            index = torch.mm(self.stateSpace, fs.T)
            joint = torch.exp(index)
            z = torch.sum(joint, dim=0)
            pMargin = torch.zeros((fs.shape[0], fs.shape[1]), dtype=torch.float64).to(device)
            for i in range(fs.shape[0]):
                for j in range(fs.shape[1]):
                    pMargin[i, j] = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.stateSpace[:, j] > 0)[0]))
            return pMargin

    def generateStateSpace(self, hierarchy, total_nodes, levels):
        stateSpace = torch.zeros(total_nodes + 1, total_nodes)
        recorded = torch.zeros(total_nodes)
        i = 1

        if levels == 2:
            for path in hierarchy:
                if recorded[path[1]] == 0:
                    stateSpace[i, path[1]] = 1
                    recorded[path[1]] = 1
                    i += 1
                stateSpace[i, path[1]] = 1
                stateSpace[i, path[0]] = 1
                i += 1

        elif levels == 3:
            for path in hierarchy:
                if recorded[path[1]] == 0:
                    stateSpace[i, path[1]] = 1
                    recorded[path[1]] = 1
                    i += 1
                if recorded[path[2]] == 0:
                    stateSpace[i, path[1]] = 1
                    stateSpace[i, path[2]] = 1
                    recorded[path[2]] = 1
                    i += 1
                stateSpace[i, path[1]] = 1
                stateSpace[i, path[2]] = 1
                stateSpace[i, path[0]] = 1
                i += 1
            
        if i == total_nodes + 1:
            return stateSpace
        else:
            print('Invalid StateSpace!!!')
