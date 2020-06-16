import pytest
import shutil
import os
from copy import deepcopy
import numpy as np
import math
import torch

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec

from corgie.boundingcube import BoundingCube
from corgie.mipless_cloudvolume import MiplessCloudVolume
from corgie.data_backends import CVDataBackend
from corgie.pairwise import PairwiseTensors 
from corgie.cli.pairwise_normalize_weights import create_z_bump, \
                                                  normalize_weights, \
                                                  PairwiseNormalizeWeightsTask

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_create_z_bump():
    shape = (5, 1, 2, 2)
    sigma = 1
    device = 'cpu'
    z_bump = create_z_bump(shape=shape, sigma=sigma, device=device)
    assert(z_bump.shape == shape)
    # symmetric over channel
    assert(torch.equal(z_bump[:,0,0,:], z_bump[:,0,1,:]))
    assert(torch.equal(z_bump[:,0,:,0], z_bump[:,0,:,1]))
    mean = 2.
    var = sigma**2
    k = np.arange(shape[0])
    v = (1. / np.sqrt(2.*math.pi*var)) * np.exp(-(k-mean)**2. / (2.*var))
    v = torch.from_numpy(v.astype(np.float32))
    assert(torch.allclose(z_bump[:,0,0,0], v))

def test_normalize_weights():
    weights = torch.zeros((5, 1, 2, 2))
    weights[1, :, :, :] = 1
    normed_weights = normalize_weights(weights=weights, sigma=1)
    assert(torch.equal(normed_weights, weights))
    normed_weights = normalize_weights(weights=weights, sigma=3)
    assert(torch.equal(normed_weights, weights))
    weights = torch.zeros((5, 1, 2, 2))
    normed_weights = normalize_weights(weights=weights, sigma=1)
    assert(torch.equal(normed_weights, weights))

def test_pairwise_normalize_weights():
    # TODO: test with CUDA
    mip=0
    sz = Vec(*[4, 4, 1])
    info = CloudVolume.create_new_info(
             num_channels=1, 
             layer_type='image', 
             data_type='float32', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,0], 
             volume_size=[8,8,4],
             chunk_size=sz,
             )
    weight_ref = MiplessCloudVolume(path='file:///tmp/cloudvolume/empty_weights',
                             info=info,
                             overwrite=True)

    def get_bcube(sz, x_block, y_block, z):
        return BoundingCube(xs=x_block*sz, xe=(x_block+1)*sz, 
                           ys=y_block*sz, ye=(y_block+1)*sz, 
                           zs=z, ze=z+1,
                           mip=mip)

    backend = CVDataBackend(device='cpu')
    offsets = [-3,-2,-1,0,1,2,3]
    IW = PairwiseTensors(name='input_weights',
                        folder='file:///tmp/cloudvolume/input_weights')
    IW.add_offsets(offsets, dtype='float32', readonly=False, reference=weight_ref)
    OW = PairwiseTensors(name='output_weights',
                        folder='file:///tmp/cloudvolume/output_weights')
    OW.add_offsets(offsets, dtype='float32', readonly=False, reference=weight_ref)
    w = torch.ones((1,1,4,4))
    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=0)
    IW.write(data=w, tgt_to_src=(1,0), bcube=bcube, mip=mip)

    normalize_task = PairwiseNormalizeWeightsTask(input_weights=IW,
                                                output_weights=OW,
                                                mip=mip,
                                                pad=0,
                                                crop=0,
                                                bcube=bcube,
                                                bump_sigma=1,
                                                fill_identity=True)
    normalize_task.execute()

    # IW[(0,1)] = 1
    # IW[(0,0)] = 1 from fill_identity=True
    def eval_gaussian(k, mean=0, var=1):
        return (1. / np.sqrt(2.*math.pi*var)) * np.exp(-(k-mean)**2. / (2.*var))
    w = torch.ones((1,1,4,4))
    den = sum([eval_gaussian(i) for i in [0,1]])
    w = torch.ones((1,1,4,4)) * (eval_gaussian(0) / den)
    ow = OW.read((0,0), bcube=bcube, mip=mip)
    assert(torch.allclose(w, ow))
    w = torch.ones((1,1,4,4)) * (eval_gaussian(1) / den)
    ow = OW.read((1,0), bcube=bcube, mip=mip)
    assert(torch.allclose(w, ow))
    
    normalize_task = PairwiseNormalizeWeightsTask(input_weights=IW,
                                                output_weights=OW,
                                                mip=mip,
                                                pad=0,
                                                crop=0,
                                                bcube=bcube,
                                                bump_sigma=1,
                                                fill_identity=False)
    normalize_task.execute()
    w = torch.zeros((1,1,4,4)) 
    ow = OW.read((0,0), bcube=bcube, mip=mip)
    assert(torch.allclose(w, ow))
    w = torch.ones((1,1,4,4))
    ow = OW.read((1,0), bcube=bcube, mip=mip)
    assert(torch.allclose(w, ow))

    delete_layer('/tmp/cloudvolume/empty_weights')
    delete_layer('/tmp/cloudvolume/input_weights')
    delete_layer('/tmp/cloudvolume/output_weights')
