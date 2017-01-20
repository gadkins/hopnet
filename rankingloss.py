import caffe
import numpy as np
import hierarchy as hier

""" 
	This Python Layer class implements the Weighted Approximate Ranking (WARP) 
	loss function as described in:

		Yunchao Gong, Yangqing Jia, Thomas Leung, Alexander Toshev, Sergey Ioffe: 
		Deep Convolutional Ranking for Multilabel Image Annotation. CVPR, 2014. 

"""
class RankingLossLayer(caffe.Layer):

	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception("Two inputs required to compute ranking loss")

	def reshape(self, bottom, top):
		if bottom[0].data[0,0].shape != bottom[1].data[0,0].shape:
			raise Exception("Inputs and labels must be of equal height and width." \
				"Instead found: %d vs. %d", \
				bottom[0].data.shape, bottom[1].data.shape)
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		top[0].reshape(1)

	def forward(self, bottom, top):
		
		self.labels = np.unique(bottom[1].data).astype(int).tolist()
		scale_factor = np.zeros_like(bottom[0].data)
		missed_margins = np.zeros_like(bottom[0].data)
		loss = 0.0
		for c in range(bottom[1].channels):
			labelset = np.unique(bottom[1].data[:,c,:,:]).astype(int).tolist()
			scale_factor[:,labelset,:,:] = 1/(c+1)
		for j in self.labels:
			J = hier.get_family(j)
			K = list(set(self.labels) - set(J))
			_,_,r,c = np.where(bottom[1].data == j)
			scores_j = bottom[0].data[:,j,r,c]
			scores_j = scores_j[np.newaxis,...]
			scores_k = bottom[0].data[:,K,:,:]
			scores_k = scores_k[:,:,r,c]
			margins = scale_factor[:,j,r,c]*np.maximum(0, 1 - scores_j + scores_k)
			loss += np.sum(margins)
			missed_margins[:,j,r,c] = -(margins > 0).astype(int).sum(1) # true class gradients
			for idx,k in enumerate(K):
				missed_margins[:,k,r,c] += (margins[:,idx,:] > 0).astype(int) # other gradients
		self.diff[...] = missed_margins
		top[0].data[...] = loss


	def backward (self, top, propagate_down, bottom):
		for i in range(2):
			if not propagate_down[i]:
				continue
			bottom[i].diff[...] = self.diff

