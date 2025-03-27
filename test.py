import torch_geometric
from torch_geometric.datasets import Planetoid, Amazon, Coauthor,Airports,GitHub
data_cora = Planetoid(root='./data',name='Pubmed')
data_amazon = Amazon(root='./data',name='Photo')
print(data_amazon)