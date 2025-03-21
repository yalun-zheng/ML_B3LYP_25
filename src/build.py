import numpy as np
import torch
torch.set_default_dtype(torch.float64)
class Node:
    def __init__(self, x, y, center, label, upper_limit, lower_limit, parent, child):
        self.x = x
        self.y = y
        self.center = center
        self.label = label
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.parent = parent
        self.child = child
        self.num = len(self.x)
    def set_child(self, child):
        self.child.append(child)
    def clear(self):
        self.x = None
        self.y = None
    

class Tree:
    def __init__(self, n_ary=2, dist_criterion=1e-5, label_criterion=1e-4, outlier_criterion=1e-13, dist_func=None, a=0.2, n_samples=11, device='cpu'):
        self.n_ary = n_ary
        self.dist_criterion = dist_criterion
        self.label_criterion = label_criterion
        self.device = device
        if dist_func is None:
            self.dist_func = self.func
        else:
            self.dist_func = dist_func
        self.outlier = []
        self.outlier_label = []
        self.outlier_criterion = outlier_criterion
        pos = np.linspace(-a/2, a/2, n_samples)
        mesh = np.meshgrid(pos,pos,pos, indexing='ij')
        mesh = np.stack(mesh, axis=-1).reshape(-1,3)
        theta = (1.35*a/(n_samples-1))**2
        self.decay = np.exp(-(mesh**2).sum(axis=-1) / theta)
        self.leaf_center = None
        self.leaf_label = None
        self.leaf_num =None
    # def func(self, points, center):
    #     return np.einsum('ij->i', np.abs(points-center))/np.einsum('ij->i',np.abs(points+center))
    def func(self, points, center):
        if type(points)!=torch.Tensor:
            return np.einsum('ij->i', np.abs(points-center)/1331)
        else:
            return torch.einsum('ij->i', torch.abs(points-center)/1331)
    # def func(self, points, center):
    #     return np.einsum('ij, j->i', np.abs(points-center)/1331, self.decay)

    def kmeans2(self, x, max_iter, chunk_size=40000):
        center = []
        labels = []
        class_num = 0
        for chunk_id in range(int(np.ceil(len(x)/chunk_size))):
            x_chunk = x[chunk_size*chunk_id:chunk_size*(chunk_id+1)]
            n_ary = min(self.n_ary, len(x_chunk))
            random_centers_id = np.random.choice(range(len(x_chunk)), size=n_ary, replace=False)
            center_chunk = x_chunk[random_centers_id]
            for _ in range(max_iter):
                if _%10==0:
                    print(f'{_} kmeans iters', end='\r')
                # distances = self.dist_func(x_chunk[:,None,:], center_chunk)
                distances = np.zeros((len(x_chunk), n_ary))
                for i, point in enumerate(x_chunk):
                    distances[i] = self.dist_func(center_chunk, point)
                labels_chunk = np.argmin(distances, axis=1)
                new_center = np.zeros_like(center_chunk)
                for i in range(n_ary):
                    cluster_points = x_chunk[labels_chunk == i]
                    if len(cluster_points) > 0:
                        new_center[i] = np.mean(cluster_points, axis=0)
                    else:
                        new_center[i] = center_chunk[i]
                if np.all(center_chunk == new_center):
                    break
                center_chunk = new_center
            center.append(center_chunk)
            labels.append(labels_chunk+class_num)
            class_num += len(set(labels_chunk))
        return np.concatenate(center, axis=0), np.concatenate(labels, axis=0)

    def kmeans(self, x, max_iter, chunk_size=40000):
        n_ary = min(self.n_ary, len(x))
        random_centers_id = np.random.choice(range(len(x)), size=n_ary, replace=False)
        center = x[random_centers_id]
        for _ in range(max_iter):
            # if _%10==0:
            #     print(f'{_} kmeans iters', end='\r')
            # distances = self.dist_func(x_chunk[:,None,:], center_chunk)
            labels = np.zeros(len(x))
            # distances = np.zeros((len(x), n_ary))
            distance = np.zeros((len(x),)) + 1e10
            for l in range(n_ary):
                new_distance = self.dist_func(x, center[l])
                labels = np.where(new_distance<distance, l, labels)
                distance = np.where(new_distance<distance, new_distance, distance)
            new_center = np.zeros_like(center)
            for i in range(n_ary):
                cluster_points = x[labels == i]
                if len(cluster_points) > 0:
                    new_center[i] = np.mean(cluster_points, axis=0)
                else:
                    new_center[i] = center[i]
            if np.all(center == new_center):
                break
            center = new_center
        return center, labels

    def fit(self, x, y, max_iter=100, most_data_num=None, least_data_num=1):
        assert len(y.shape) == 1
        if most_data_num is None:
            most_data_num = len(y)
        root = Node(x=x, y=y, center=None, label=np.array(0.), upper_limit=np.array(1e10), lower_limit=np.array(-1e10), parent=None, child=[])
        self.layers = [[root]]
        while True:
            layer = []
            for parent in self.layers[-1]:
                if len(parent.y) <= least_data_num:
                    parent.clear()
                    continue
                if (parent.upper_limit-parent.label<self.label_criterion).all() and (parent.label-parent.lower_limit<self.label_criterion).all() and (len(parent.y) <= most_data_num):
                    parent.clear()
                    continue
                center, labels = self.kmeans(parent.x, max_iter)
                while len(set(labels)) == 1:
                    center, labels = self.kmeans(parent.x, max_iter)
                for i in range(len(set(labels))):
                    cluster_x = parent.x[labels == i]
                    cluster_y = parent.y[labels == i]
                    if len(cluster_x):
                        y_mean = np.mean(cluster_y, axis=0)
                        y_min = np.min(cluster_y, axis=0)
                        y_max = np.max(cluster_y, axis=0)
                        # Prof. Chen's advice: center and label are better to be a pair from original data, rather than an average
                        # min_diff_id = np.argmin(np.einsum('ij->i', np.abs(cluster_x - center[i])))
                        # node = Node(cluster_x, cluster_y, center=cluster_x[min_diff_id], label=cluster_y[min_diff_id], upper_limit=y_max, lower_limit=y_min, parent=parent, child=[])
                        node = Node(cluster_x, cluster_y, center=center[i], label=y_mean, upper_limit=y_max, lower_limit=y_min, parent=parent, child=[])
                        parent.set_child(node)
                        layer.append(node)
                parent.clear()
            if layer:
                self.layers.append(layer)
                print(f'{len(self.layers):10d} layers, {len(layer):10d} nodes in current layer', end='\r')
            else:
                print()
                break
        self.leaf_center, self.leaf_label, self.leaf_num = self.get_center_label_num()
        print(len(self.leaf_num), 'classes')
            
    def inference(self, x, dist_criterion=None, return_dist=False):
        if self.outlier:
            dist_from_outlier = self.dist_func(x, np.array(self.outlier))
            min_dist_id = np.argmin(dist_from_outlier)
            if dist_from_outlier[min_dist_id] <= self.outlier_criterion:
                return self.outlier_label[min_dist_id]
        root = self.layers[0][0]
        while root.child:#TODO or 
            dists = self.dist_func([root.child[0].center, root.child[1].center] , x)
            # dists = [self.dist_func(child.center[None,:], x) for child in root.child]
            min_dist_id = np.argmin(dists)
            y = root.child[min_dist_id].label
            root = root.child[min_dist_id]
        if dist_criterion is not None and dists[min_dist_id]>dist_criterion:
            y = y*np.NaN
        return (y, dists[min_dist_id]) if return_dist else y

    def inference2_old(self, x, dist_criterion=None):
        if self.outlier:
            dist_from_outlier = self.dist_func(x, np.array(self.outlier))
            min_dist_id = np.argmin(dist_from_outlier)
            if dist_from_outlier[min_dist_id] <= self.outlier_criterion:
                return self.outlier_label[min_dist_id]
        dists = self.dist_func(self.leaf_center, torch.tensor(x, device=self.device))
        min_dist_id = torch.argmin(dists)
        y = self.leaf_label[min_dist_id]
        if dist_criterion is not None and dists[min_dist_id]>dist_criterion:
            y = y*np.NaN
        return y
    def inference2(self, xs, batch_size=200):
        with torch.no_grad():
            xs = torch.tensor(xs, device=self.device)
            batch_num = len(xs)//batch_size
            ys = torch.zeros(len(xs), device=self.device)
            
            for batch_id in range(batch_num):
                batch_xs = xs[batch_id*batch_num:(1+batch_id)*batch_num]
                dist = torch.einsum('nma->nm', torch.abs(self.leaf_center[:,None,:]-batch_xs)/1331)
                min_dist_id = torch.argmin(dist, dim=0)
                ys[batch_id*batch_num:(1+batch_id)*batch_num] = self.leaf_label[min_dist_id]
            if len(xs) > (1+batch_id)*batch_num:
                batch_xs = xs[(1+batch_id)*batch_num:]
                dist = torch.einsum('nma->nm', torch.abs(self.leaf_center[:,None,:]-batch_xs)/1331)
                min_dist_id = torch.argmin(dist, dim=0)
                ys[(1+batch_id)*batch_num:] = self.leaf_label[min_dist_id]
            return ys.cpu().numpy()

    def set_outlier(self, x, y):
        self.outlier.append(x)
        self.outlier_label.append(y)
    
    def get_center_label_num(self):
        center = []
        label = []
        num = []
        for layer in self.layers:
            for node in layer:
                if not node.child:
                    center.append(node.center)
                    label.append(node.label)
                    num.append(node.num)
        return torch.tensor(np.array(center), device=self.device), torch.tensor(np.array(label), device=self.device), np.array(num)

if __name__ == '__main__':
    pass
    