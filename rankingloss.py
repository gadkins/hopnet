import caffe
import numpy as np
from hierarchy import get_parents

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
        L = np.unique(bottom[1].data).astype(int)
        Y = bottom[1].data
        scores = bottom[0].data
        self.missed_margins = np.zeros_like(bottom[0].data)
        loss = 0.0

    	#for each label
            #(1) in bottom[1], find indices matching label y (e.g. all "background" indices)
            #(2) in bottom[0] on label page y, get scores at those indices (s_y)
            #(4) get scores s_j at those indices on parent pages (J)
            #(3) get scores s_k at those indices on nonparent pages (K)
		for y in L:
			_,_,r,c = np.where(Y == y)
			s_y = scores[:,y,r,c]
			s_y = s_y[np.newaxis,...]
			J = get_parents(tree, y)
			if J:
				s_j = scores[:,J,:,:]
				s_j = s_j[:,:,r,c]
				loss_j = np.maximum(0, 1 - s_j + s_y)
				loss += np.sum(loss_j)
				# parent classes promoted with extra negative gradient
				self.missed_margins[:,y,r,c] += -(loss_j > 0).astype(int).sum(1)
			K = list(set(L) - set(J) - set([y]))
			if K:
				s_k = scores[:,K,:,:]
				s_k = s_k[:,:,r,c]
				loss_k = np.maximum(0, 1 - s_y + s_k)
				loss += np.sum(loss_k)
				self.missed_margins[:,y,r,c] += -(loss_k > 0).astype(int).sum(1) # true class gradients
				for idx,k in enumerate(K):
					self.missed_margins[:,k,r,c] += (loss_k[:,idx,:] > 0).astype(int) # other gradients

        top[0].data[...] = loss
        

    def backward (self, top, propagate_down, bottom):
        # grad = {    0   if y*np.dot(w,x) >= 1
        #          -y*x   else   } 
        # where x is element of inputs X and y is element of labels Y
        # here Y = bottom[0].data and X = self.inputs
        for i in range(2):
            if not propagate_down[i]:
                continue
            self.missed_margins[self.missed_margins > 0] = 0
            bottom[i].diff[...] = bottom[0].data * self.missed_margins

