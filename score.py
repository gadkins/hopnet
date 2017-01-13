from __future__ import division
import caffe
import numpy as np
import os, sys
from datetime import datetime
from PIL import Image

# Assumes sequential labels with minimum label value (offset) of "o"
def fast_hist(a, b, n, o):
	k = (a >= o) & (a < n + o)
	return np.bincount(n * (a[k].astype(int)-o) + b[k] - o, minlength=n**2).reshape(n, n)

def compute_hist(net, rank_dir, dataset, layer='score', gts=['label'], n_cls=[2, 7, 25]):
	if rank_dir and not os.path.exists(rank_dir): 
		os.mkdir(rank_dir)
	loss = 0
	offsets = [0]
	hists = []
	for k,v in enumerate(n_cls):
		if k < len(n_cls)-1:
			offsets.append(v + sum(offsets))
		hists.append(np.zeros((v, v)))
	for idx in dataset:
		net.forward()
		if rank_dir:
			# find top-k ranks
			ranks = collect_ranks(net.blobs[layer].data)
			# compute histogram
			for k,v in enumerate(n_cls):
				hists[k] += (fast_hist(net.blobs[gts[0]].data[:,k,:,:].flatten(),
							ranks[k].flatten(), v, offsets[k]))
			# save output images
			for k,rank in enumerate(ranks):
				rank -= offsets[k]
				im = Image.fromarray(rank.astype(np.uint8), mode='P')
				im.save(os.path.join(rank_dir, 'class{}'.format(k), idx + '.png'))
	# compute the loss
	loss += net.blobs['loss'].data.flat[0] / len(dataset)
	return hists, loss

""" Given an 4D prediction array, finds the page (axis=1) on which the largest element 
	at each element position was found. Page ranks are in descending order. """
def collect_ranks(prediction):
	rank = np.argsort(prediction, axis=1)[:,::-1,:,:] # highest to lowest
	return [np.squeeze(rank[:,k,:,:]) for k in range(3)] # top-k labelmaps

def seg_tests(solver, rank_format, dataset, layer, gts, n_cls):
	print '>>>', datetime.now(), 'Begin seg tests'
	solver.test_nets[0].share_with(solver.net)
	do_seg_tests(solver.test_nets[0], solver.iter, sm_format, rank_format, dataset, layer, gts)

def do_seg_tests(net, iter, rank_format, dataset, layer, gts, n_cls):
	if rank_format:
		rank_format = rank_format.format(iter)
	hists, loss = compute_hist(net, rank_format, dataset, layer, gts, n_cls)
	# Calculate metrics
	tp = fn = fp = acc = iu = freq = prec = recall = 0
	for i, hist in enumerate(hists):
		# mean loss
		print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
		# true positives, false negatives, false positives
		tp = np.diag(hist)
		fn = hist.sum(0) - tp
		fp = hist.sum(1) - tp
		# overall accuracy
		acc = tp.sum() / hist.sum()
		print '>>>', datetime.now(), 'Iteration', iter, \
			  'overall accuracy label{}'.format(i), acc
		# per-class IU
		iu = tp / (tp + fp + fn)
		print '>>>', datetime.now(), 'Iteration', iter, 'mean IU label{}'.format(i), \
			  np.nanmean(iu)
		# frequency
		freq = hist.sum(1) / hist.sum()
		print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc label{}'.format(i), \
			  (freq[freq > 0] * iu[freq > 0]).sum()
		# precision
		prec = tp / (tp + fp)
		print '>>>', datetime.now(), 'Iteration', iter, 'mean precision label{}'.format(i), \
			  np.nanmean(prec)
		# recall
		recall = tp / (tp + fn)
		print '>>>', datetime.now(), 'Iteration', iter, 'mean recall label{}'.format(i), \
			  np.nanmean(recall), '\n'
		print '-------------------------------------------------------------------'

	return hists
