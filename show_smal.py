# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:05:12 2022

@author: Chengguo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pickle as pkl
from vispy import scene, io
from plyfile import *
from mpl_toolkits.mplot3d import Axes3D

base_dir = "C:/Users/Chengguo/Desktop/smpl_model/"
model_file_name = "my_smpl_00781_4_all.pkl"
data_file_name = "my_smpl_data_00781_4_all.pkl"
uv_file_name = "my_smpl_00781_4_all_template_w_tex_uv_001.pkl"
sym_file_name = "symIdx.pkl"

with open(base_dir + model_file_name, 'rb') as f:
     u = pkl._Unpickler(f)
     u.encoding = "latin1"
     dd = u.load()

print(dd.keys())

with open(base_dir + data_file_name, 'rb') as f:
     u = pkl._Unpickler(f)
     u.encoding = "latin1"
     kk = u.load()

print(kk.keys())

with open(base_dir + uv_file_name, 'rb') as f:
     u = pkl._Unpickler(f)
     u.encoding = "latin1"
     vv = u.load()

print(vv.keys())


def vis_other_smal(dd,kk):
    
    clusters = ["cat","wolf","horse","cow","hippo"]
    for i in range(len(kk['cluster_means'])):
        plt.figure(figsize=(10,8))
        betas = kk['cluster_means'][i]
    # for i in range(len(kk['toys_betas'])):
    #     betas = kk['toys_betas'][i]
        
        v_template = dd["v_template"] + np.matmul(dd["shapedirs"],betas)
        X = v_template[:,0]
        Y = v_template[:,1]
        Z = v_template[:,2]
        

        plt.ion()   
        ax = plt.axes(projection="3d")
        ax.scatter3D(X,Y,Z,s=20)
        
        # for m in dd['f']:###3角网格的索引,[1796,1797,33]
        #     x,y,z=[],[],[]
        #     for j in m:
        #         x.append(X[j])
        #         y.append(Y[j])
        #         z.append(Z[j])
        #     ax.plot3D(x,y,z)
        
        plt.rcParams.update({'font.family': 'Times New Roman'})
        plt.rcParams.update({'font.weight': 'normal'})
        plt.rcParams.update({'font.size': 20})
        plt.xlabel('X')
        plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
        ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）    
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        plt.title(f"the {i} model")
        plt.title(f"{clusters[i]}")
        plt.pause(2)
        
    plt.ioff()
    plt.close()

vis_other_smal(dd,kk)
        


def vis_shapedirs(dd):
    X1 = dd["v_template"][:,0]
    Y1 = dd["v_template"][:,1]
    Z1 = dd["v_template"][:,2]
    
    plt.figure(figsize=(10,8))
    for i in range(41):
        X = dd["shapedirs"][:,:,i][:,0]+X1
        Y = dd["shapedirs"][:,:,i][:,1]+Y1
        Z = dd["shapedirs"][:,:,i][:,2]+Z1
        
        ax = plt.axes(projection="3d")
        ax.scatter3D(X,Y,Z,s=20)
        plt.ion()        
              
                 
        plt.rcParams.update({'font.family': 'Times New Roman'})
        plt.rcParams.update({'font.weight': 'normal'})
        plt.rcParams.update({'font.size': 20})
        plt.xlabel('X')
        plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
        ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）    
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.title(f"the {i} model")
        
        plt.pause(1)
        
        plt.show()
    plt.ioff()
    plt.close()


# vis_shapedirs(dd)

def show_verts_joints(dd, vis_joints= False, kintree_table = False, vis_weights = False):
    X = dd["v_template"][:,0]
    Y = dd["v_template"][:,1]
    Z = dd["v_template"][:,2]
    
    plt.figure(figsize=(10,8))
    ax = plt.axes(projection="3d")
    # ax.scatter3D(X,Y,Z,s=5,alpha=0.5)
    k=0
    for a1, b1, c1 in zip(X, Y, Z):
        # print(a1[0][0], b1[0][0], c1[0][0]+0.01)
        ax.text(float(a1), float(b1), float(c1) + 0.01, '%s' % k, ha='center', va='bottom', fontsize=15)
        k=k+1
        if k > 100:
            break
    
    if vis_weights == True:
        colors = np.zeros_like(dd["weights"][:,0])
        for i in range(35):
            vert_weigths = dd["weights"][:,i]
            if np.all(vert_weigths==0):
                print("weights------------")
            colors[vert_weigths!=0] = i+10
        
        
        ax.scatter3D(X,Y,Z,c=colors,s=20)
    
    if vis_joints == True:
        joints = np.matmul(dd["J_regressor"].todense(),dd["v_template"])
        
        X = joints[:,0] 
        Y = joints[:,1]
        Z = joints[:,2]
        
        ax.scatter3D(X,Y,Z,c="r",s=50)
        K=0
        for a1, b1, c1 in zip(X, Y, Z):
            # print(a1[0][0], b1[0][0], c1[0][0]+0.01)
            ax.text(float(a1), float(b1), float(c1)+0.01, '%s'%K, ha='center', va='bottom', fontsize=9)
            K=K+1
    
    
    if kintree_table == True:
        for i in dd["kintree_table"].T:
            if i[0] > 35:
                i = [0,0]
            x,y,z=[],[],[]
            # print(i)
            for j in i:
                # print(j)
                x.append(float(joints[j][:,0]))
                y.append(float(joints[j][:,1]))
                z.append(float(joints[j][:,2]))
            # print(x,y,z)
            # print("------------")          
            ax.plot3D(x,y,z)
        
    
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'font.size': 20})
    plt.xlabel('X')
    plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    plt.show()

    # x = MultipleLocator(1)    # x轴每10一个刻度
    # y = MultipleLocator(1)    # y轴每15一个刻度
    # # 设置刻度间隔
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x)
    # ax.yaxis.set_major_locator(y)


def show_mesh(dd):
    plt.figure(figsize=(12,10))
    axes = plt.gca(projection='3d')
    for i in dd['f']:###3角网格的索引,[1796,1797,33]
        x,y,z=[],[],[]
        for j in i:
            x.append(dd["v_template"][:,0][j])
            y.append(dd["v_template"][:,1][j])
            z.append(dd["v_template"][:,2][j])
        axes.plot3D(x,y,z)
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    axes.set_zlim(-1, 1)
    
    
    
def show_joints(dd):
    plt.figure(figsize=(12,10))
    ax = plt.gca(projection='3d')
    joints = np.matmul(dd["J_regressor"].todense(),dd["v_template"])
    
    X = joints[:,0]
    Y = joints[:,1]
    Z = joints[:,2]
    
    ax.scatter3D(X,Y,Z)
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'font.size': 20})
    plt.xlabel('X')
    plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)


# plt.savefig('3D.jpg', bbox_inches='tight', dpi=2400)  # 保存图片，如果不设置 bbox_inches='tight'，保存的图片有可能显示不全
# plt.show()
# plt.pause(5)

# show_verts_joints(dd,vis_joints=True, kintree_table = True, vis_weights=True)

def vis_obj():
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    
    verts, faces, normals, nothing = io.read_mesh("C:/Users/Chengguo/Desktop/111/pig.obj")
    
    mesh = scene.visuals.Mesh(vertices=verts, faces=faces, shading='smooth')
    
    view.add(mesh)
    
    view.camera = scene.TurntableCamera()
    view.camera.depth_value = 10


def align_smal_template_to_symmetry_axis(v, sym_file):
    # These are the indexes of the points that are on the symmetry axis
    I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964, 975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]

    v = v - np.mean(v)
    y = np.mean(v[I,1])
    v[:,1] = v[:,1] - y
    # v[I,1] = 0

    # symIdx = pkl.load(open(sym_path))
    with open(sym_file, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        symIdx = u.load()

    
    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    v[left[symIdx]] = np.array([1,-1,1])*v[left]

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    try:
        assert(len(left_inds) == len(right_inds))
    except:
        import pdb; pdb.set_trace()

    return v, left_inds, right_inds, center_inds



def vis_ply():
    plydata = PlyData.read('C:/Users/Chengguo/Desktop/pig.ply')
    
    xlist = (plydata['vertex']['x']-min(plydata['vertex']['x']))/(max(plydata['vertex']['x'])-min(plydata['vertex']['x']))-0.5
    ylist = (plydata['vertex']['y']-min(plydata['vertex']['y']))/(max(plydata['vertex']['y'])-min(plydata['vertex']['y']))-0.5
    zlist = (plydata['vertex']['z']-min(plydata['vertex']['z']))/(max(plydata['vertex']['z'])-min(plydata['vertex']['z']))-0.5
       
    fig = plt.figure(figsize=(16, 14))
    ax = Axes3D(fig)

    xlist1 = zlist
    ylist1 = xlist
    zlist1 = ylist
    
    v_sym, left_inds, right_inds, center_inds = \
    align_smal_template_to_symmetry_axis(
        np.array([xlist1,ylist1,zlist1]).reshape(3889,3), base_dir+sym_file_name)
    
    xlist1 = v_sym[:,0]
    ylist1 = v_sym[:,1]
    zlist1 = v_sym[:,2]

    k=0
    for a1, b1, c1 in zip(xlist1, ylist1, zlist1):
        # print(a1[0][0], b1[0][0], c1[0][0]+0.01)
        # if k == 26:
        ax.text(float(a1), float(b1), float(c1) + 0.01, '%s' % k, ha='center', va='bottom', fontsize=8)
        if k > 20:
            break
        k=k+1


    # for i in range(len(plydata['face'])):
    #     x, y, z = [], [], []
    #     for j in plydata['face'][i][0]:
    #
    #         x.append(xlist1[j])
    #         y.append(ylist1[j])
    #         z.append(zlist1[j])
    #     ax.plot3D(x, y, z)

    ax.scatter3D(xlist1, ylist1, zlist1)

    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'font.size': 20})

    plt.xlabel('X')
    plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
    
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    plt.show()##

# vis_obj()
# vis_ply()

# if __name__ == '__main__':
#     canvas.app.run()




















