from datautils.describe_visualise import *

# if __name__ == '__main__':
#     config.data_name = "UMIST"
#     config.save_path = r"G:\File\GitHub\Deep_Graph_Representation_via_Adaptive_Homotopy_Contrastive_Learning\logs\umist_1024\test1"
#     G = nx.Graph()
#     for skip in range(0, 10):
#         epoch = skip
#         kk = 2
#         Z, Y, T = gen_visual(epoch,kk)

#         node_pos, color_idx = plot_embeddings(Z, T, 'G', epoch)

#         myWeight = []
#         myNode = np.argwhere(Y > 0.6)
#         edges = []
#         for i in range(myNode.shape[0]):
#             G.add_edge(myNode[i][0], myNode[i][1],
#                        weight=Y[myNode[i][0], myNode[i][1]])

#             edges.append(myNode[i])
#             myWeight.append(Y[myNode[i][0], myNode[i][1]]**4)

#         # 按权重划分为重权值得边和轻权值的边
#         elarge = [(u, v)
#                   for (u, v, d) in G.edges(data=True) if d['weight'] > 0.9]
#         esmall = [(u, v)
#                   for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.9]
#         # 打开交互模式
#         fig = plt.gcf()
#         fig.set_size_inches(14, 12, forward=True)
#         plt.title(config.data_name+"-Y-{}".format(int(epoch*kk)))
#         plt.ion()
#         # plt.axis('equal')
#         plt.axis('off')
#         fig.tight_layout()  # 调整整体空白
#         # 设置坐标轴范围
#         # plt.xlim((-5, 5))
#         # plt.ylim((-2, 2))

#         G.add_nodes_from(np.arange(0, len(T)))
#         len_edges = int(len(edges)/20)
#         k = 0
#         # plot
#         nx.draw_networkx_nodes(
#             G, pos=node_pos, node_size=0.5, alpha=0.3, node_color=T)
#         im = nx.draw_networkx_edges(G, pos=node_pos, edgelist=edges, width=1,
#                                alpha=0.1, edge_color=myWeight, edge_cmap=plt.cm.rainbow)
#         # 增加右侧的颜色刻度条
#         plt.colorbar(im)    
#         plt.savefig(r"D:\Desktop\1\kt" + "/{}-0200.png".format(epoch))
#         for c, idx in color_idx.items():
#             # 根据权重
#             plt.scatter(node_pos[idx, 0], node_pos[idx, 1],
#                         label=c, s=40,marker = 'o',alpha=1)  # c=node_colors)

#             plt.savefig(r"D:\Desktop\1\kt" + "/{}-021".format(epoch) +
#                         "{}.png".format(str(c)))
#             plt.pause(0.001)

#         plt.clf()
#         plt.ioff

if __name__ == '__main__':
    plt.ion()
    for skip in range(3, 30):
        epoch = skip
        kk = 10
        try:
            Z, Y, T = gen_visual(epoch * kk, 1)
        except:
            pass
    plt.ioff
