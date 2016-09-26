import caffe
import numpy as np

        
class MultilabelSoftmaxLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Two inputs required to compute multilabel softmax loss")

    def reshape(self, bottom, top):
        if bottom[0].data.shape != bottom[1].data.shape
            raise Exception("Inputs must have the same dimension.")
        top[0].reshape(1)

    def forward(self, bottom, top):	
        probs = np.zeros_like(bottom[0].data)
        L = [np.unique(bottom[1].data[:,lm,:,:]).astype(int).tolist() 
        	for lm in range(bottom[1].channels)]
        numer = np.exp(bottom[0].data)
        for l in L:
            denom = np.sum(numer[:,l,:,:], axis=1, keepdims=True)
            probs[:,l,:,:] = numer[:,l,:,:]/denom
            top[0].data[1,:,1,1] = -(1/bottom[0].num)*np.sum(np.log(probs[:,l,:,:]))
        self.diff[...] = probs

    def backward (self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                delta[range(bottom[0].num), np.array(bottom[1].data,dtype=np.uint16)] -= 1
        bottom[i].diff[...] = delta/bottom[0].num
        
                
