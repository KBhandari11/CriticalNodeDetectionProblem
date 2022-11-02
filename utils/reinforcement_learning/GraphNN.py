import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

class GraphNN(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               feature_size,
               hidden_sizes,global_size):
    """GNN.
    Args:
      feature_size: (int) number of centrality features
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      global_size: (int) number of global feature
    """
    super(GraphNN, self).__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_conv, hidden_global, hidden_final = hidden_sizes[0], hidden_sizes[1], hidden_sizes[2]
    
    #2 GAT layer
    self.conv1 = GATv2Conv(feature_size, hidden_conv[0],add_self_loops = True) 
    self.conv1bn = gnn.BatchNorm(hidden_conv[0])
    self.conv2 = GATv2Conv(hidden_conv[0],hidden_conv[1],add_self_loops = True) 
    self.conv2bn = gnn.BatchNorm(hidden_conv[1])
    
    #Global_Linear Layer
    self.linear_global1 = nn.Linear(global_size, hidden_global[0])
    self.linear_global2 = nn.Linear(hidden_global[0], hidden_global[1])
    
    
    #Join to Layer
    self.linear1 = nn.Linear(hidden_conv[-1]+hidden_global[-1],hidden_final[0])
    self.batchnorm1 = nn.BatchNorm1d(hidden_final[0])
    self.linear2 = nn.Linear(hidden_final[0], hidden_final[1])
    self.batchnorm2 = nn.BatchNorm1d(hidden_final[1])
    self.linear3 = nn.Linear(hidden_final[1], 1)
  
  def forward(self, node_feature, edge_index, global_x):
        
    x, edge_index,global_x = node_feature.to(self.device), edge_index.to(self.device),global_x.to(self.device)
    #MLP for Global Features
    global_x = self.linear_global1(global_x)
    global_x = self.linear_global2(global_x)
    global_x = global_x.repeat(x.size()[0],1)
    
    #GAT Layer for Graphs
    x = F.relu(self.conv1(x, edge_index))
    x = self.conv1bn(x)
    x = F.relu(self.conv2(x, edge_index))
    x = self.conv2bn(x)
    
    #Combining Graph Feature Embedding and GAT Layer 
    x = F.relu(self.linear1(torch.hstack((x,global_x))))
    x = self.batchnorm1(x)
    x = F.relu(self.linear2(x))
    x = self.batchnorm2(x)
    x = F.relu(self.linear3(x))
    x = torch.softmax(x,dim=0)
    return x