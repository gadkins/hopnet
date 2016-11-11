import caffe
import numpy as np

class RankingLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Two inputs required to compute ranking loss")

    def reshape(self, bottom, top):
        if bottom[0].data[0,0].shape != bottom[1].data[0,0].shape:
            raise Exception("Inputs and labels must be of equal height and width." \
			    "Instead found: %d vs. %d", \
                            bottom[0].data.shape, bottom[1].data.shape)
        top[0].reshape(1)

    def forward(self, bottom, top):
	labelsets = tuple(tuple(np.unique(bottom[1].data[:,c,:,:]).astype(int)
            for c in range(bottom[1].channels)))
        # Given label, returns the index of the labelset to which it belongs
        def labelset_idx(label):
            return next((i for i, sub in enumerate(labelsets) if label in sub), -1)
        # Given a labelset index, returns a list of nonparent labels including itself
        def nonparent_labels(idx):
            return [item for sublist in labelsets[idx:] for item in sublist]
        loss = 0 
#       for each labelmap
#           for each label
#               (1) find indices in labelmap matching that label (e.g. all "background" indices)
#               (2) go to prediction's label page and get activation values at those indices ( fj(xi) )
#               (3) get activation values at those same indices at the nonparent pages ( fk(xi) )
#               (4) get activation values at those indices at the parent pages

        for i, ls in enumerate(labelsets):
            nonparents = nonparent_labels(i)
            for label in ls:
                nonparents.remove(label)
                if not nonparents:
                    continue
                # (1) Find indices matching label
                indices = np.where(bottom[1].data == label)
                # need to get indices in the form of [(n,c,h,w), (n,c,h,w),...] instead of (array([n,n]), array([c,c]), array([h,h]),...
                indices = (i[0] for i in indices] # needs work
                indices = tuple(item for sub in indices for item in sub)
                # (2) Case 1: label k does not belong to set of parents of label j
                loss += max(0, 1 - bottom[0].data[:,label,:,:] + np.sum(bottom[0].data[:,nonparents,:,:])
                nonparents.add(label)
                # Case 2: label k does belong to set of parents of label j
                parents = list(set(labelsets) - set(nonparents))
                loss + = max(0, 1 - np.sum(bottom[0].data[:,parents,:,:]) + bottom[0].data[:,label,:,:])
        n,c,h,w = bottom[0].data.shape
        rank = np.argsort(bottom[0].data, axis=1)
        ranks = [np.where(rank == r)[1].reshape(n,1,h,w) for r in range(3)] # 3 label maps
        self.diff[...] = 
        top[0].data[...] = loss
        

    def backward (self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


