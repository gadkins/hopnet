from __future__ import division
import caffe
import numpy as np
import hierarchy as hier

""" This Python Layer class implements the multilabel softmax loss as described in:

        Yunchao Gong, Yangqing Jia, Thomas Leung, Alexander Toshev, Sergey Ioffe: 
        Deep Convolutional Ranking for Multilabel Image Annotation. CVPR, 2014.

"""
class MultilabelSoftmaxWithLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Two inputs required to compute multilabel softmax loss")

    def reshape(self, bottom, top):
        if bottom[0].data[0,0].shape != bottom[1].data[0,0].shape:
            raise Exception("Inputs and labels must be of equal height and width." +
                            "Instead found: bottom[0] = %d and bottom[1] = %d", 
                            bottom[0].data.shape, bottom[1].data.shape)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):	
        labels = np.unique(bottom[1].data).astype(int).tolist()
        Y = np.zeros_like(bottom[0].data) # a truth mask volume
        probs = np.zeros_like(bottom[0].data)
        scale_factor = np.zeros_like(bottom[0].data)
        for l in labels:
            _,_,r,c = np.where(bottom[1].data == l)
            Y[:,l,r,c] = 1
        for c in range(bottom[1].channels):
            labelset = np.unique(bottom[1].data[:,c,:,:]).astype(int).tolist()
            scale_factor[:,labelset,:,:] = 1/(c+1)
        Y = Y.flatten().astype(int)
        exp_scores = np.exp(bottom[0].data)
        probs = (exp_scores / np.sum(exp_scores, axis=1, keepdims=True)).flatten()
        correct = -scale_factor.flatten()*np.log(probs[Y])
        self.diff = correct.reshape((bottom[0].data.shape))
        top[0].data[...] = np.sum(correct)/bottom[0].num # loss

    def backward (self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            self.diff -= 1
            bottom[i].diff[...] = self.diff / bottom[0].num
