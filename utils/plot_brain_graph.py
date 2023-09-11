import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

loc_file = open('./locfile.txt')
x = []
y = []
z = []
channel_name = []
for channel in loc_file.readlines():
    info_list = channel.split('\t')
    x.append(float(info_list[1].strip()))
    y.append(float(info_list[2].strip()))
    z.append(float(info_list[3].strip()))
    channel_name.append(info_list[4].strip())

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
for i in range(len(x)):
    ax.text(x[i],y[i],z[i],channel_name[i])


#same_adj
# adj = torch.load('./model/compare_graph/s22same_adj/model.pth', map_location=torch.device('cpu'))['state_dict']['A'].numpy()
#
# #LDG
# adj_name_list = ['bottle_neck.A','blocks.0.block1.A','blocks.0.block2.A','blocks.0.block3.A','blocks.1.block1.A',
#                  'blocks.1.block2.A','blocks.1.block3.A']
adj = torch.load('./model/compare_graph/s22LDG/model.pth', map_location=torch.device('cpu'))['state_dict']['blocks.1.block3.A'].numpy()



flat_adj = adj.flatten()
top_k = np.argsort(flat_adj)[:20]
top_x = top_k // 62
top_y = top_k % 62
sparse_adjcent = np.zeros((62, 62), dtype=np.float32)
for i in range(top_x.shape[0]):
    sparse_adjcent[top_x[i], top_y[i]] = adj[top_x[i], top_y[i]]

for i in range(top_x.shape[0]):
    ax.plot((x[top_x[i]], x[top_y[i]]), (y[top_x[i]], y[top_y[i]]), (z[top_x[i]], z[top_y[i]]))

ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('X', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('Y', fontdict={'size': 15, 'color': 'red'})
plt.show()
