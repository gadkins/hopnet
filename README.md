# HOPNet
<strong>Deep Multilabel FCN for Hierarchical Object Parsing</strong>

HOPNet is a deep fully convolutional network (FCN) modified for multiclass-multilabel problems. It utilizes the Caffe deep learning framework and is particularly well-suited for parsing objects into prediction map hierarchies.   

Instead of the typical 2D label maps utilized by FCNs, HOPNet accepts a stack of label maps as a multidimensional array of labels. During training of the network, either a multilabel softmax loss or a pairwise ranking loss function can be applied that encourages the positive samples to rank higher than negative samples and high-level hierarchies to outrank low-level. 

If used for object parsing, each of the <i>N</i> label maps in the stack can represent subsequently finer-grained hierarchical segmentations. The network will return pixel-wise predictions ranked according to the range of label values. The <i>k</i>th prediction map can be formed by collecting the <i>k</i>th predition at each pixel where <i>k</i>&#8804;<i>N</i>.

For a more detailed description of the aforementioned technique and proposed use case, please see http://www.graysonadkins.com/research
