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
            #return non.remove(label)
        def parent_labels(idx):
        loss = 0 
        
        for i, ls in enumerate(labelsets):
            nonparents = nonparent_labels(i)
            for label in ls:
                nonparents.remove(label)
                if not nonparents:
                    continue
                # Case 1: label k does not belong to set of parents of label j
                loss += max(0, 1 - bottom[0].data[:,label,:,:] + np.sum(bottom[0].data[:,nonparents,:,:])
                nonparents.add(label)
                # Case 2: label k does belong to set of parents of label j
                loss + = max(0, 1 - np.sum(bottom[0].data[:,parents,:,:]) + bottom[0].data[:,label,:,:]
        
        

    def backward (self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                delta[range(bottom[0].num), np.array(bottom[1].data,dtype=np.uint16)] -= 1
        bottom[i].diff[...] = delta/bottom[0].num
