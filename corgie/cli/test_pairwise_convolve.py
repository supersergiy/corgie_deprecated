import pytest
import shutil
import os
from copy import deepcopy
import numpy as np
import torch
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec

from corgie.boundingcube import BoundingCube
from corgie.mipless_cloudvolume import MiplessCloudVolume
from corgie.data_backends import CVDataBackend
from corgie.pairwise import PairwiseTensors, PairwiseFields
from corgie.cli.pairwise_convolve import PairwiseDotProductTask

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_pairwise_dot_product_task():
    # TODO: test with CUDA
    mip=0
    sz = Vec(*[4, 4, 1])
    field_info = CloudVolume.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='float32', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,0], 
             volume_size=[8,8,4],
             chunk_size=sz,
             )
    field_ref = MiplessCloudVolume(path='file:///tmp/cloudvolume/empty_fields',
                             info=field_info,
                             overwrite=True)

    weight_info = deepcopy(field_info)
    weight_info['num_channels'] = 1
    weight_ref = MiplessCloudVolume(path='file:///tmp/cloudvolume/empty_weights',
                             info=weight_info,
                             overwrite=True)

    def get_bcube(sz, x_block, y_block, z):
        return BoundingCube(xs=x_block*sz, xe=(x_block+1)*sz, 
                           ys=y_block*sz, ye=(y_block+1)*sz, 
                           zs=z, ze=z+1,
                           mip=mip)

    backend = CVDataBackend(device='cpu')
    offsets = [-1,0,1]
    F  = PairwiseFields(name='fields',
                        folder='file:///tmp/cloudvolume/fields')
    F.add_offsets(offsets, readonly=False, reference=field_ref)
    W = PairwiseTensors(name='weights',
                       folder='file:///tmp/cloudvolume/weights')
    W.add_offsets(offsets, dtype='float32', readonly=False, reference=weight_ref)
    CF = PairwiseFields(name='convolved_fields',
                        folder='file:///tmp/cloudvolume/convolved_fields')
    CF.add_offsets(offsets, readonly=False, reference=F)

    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=0)
    obcube = get_bcube(sz=4, x_block=0, y_block=1, z=0)
    # F[(-1, 0)] = (0,2)
    f = torch.zeros((1,2,4,4))
    f[:,1,:,:] = 2
    F.write(data=f, tgt_to_src=(-1,0), bcube=bcube, mip=mip)
    F.write(data=f, tgt_to_src=(-1,0), bcube=obcube, mip=mip)
    # F[(1, 0)] = (0,4)
    f[:,0,:,:] = 0
    f[:,1,:,:] = 4
    F.write(data=f, tgt_to_src=(1,0), bcube=bcube, mip=mip)
    F.write(data=f, tgt_to_src=(1,0), bcube=obcube, mip=mip)
    # W[(-1, 0)] = 0.375 
    w = torch.ones((1,1,4,4)) * 0.375
    W.write(data=w, tgt_to_src=(-1,0), bcube=bcube, mip=mip)
    W.write(data=w, tgt_to_src=(-1,0), bcube=obcube, mip=mip)
    # W[(0, 0)] = 0.5
    w = torch.ones((1,1,4,4)) * 0.5
    W.write(data=w, tgt_to_src=(0,0), bcube=bcube, mip=mip)
    W.write(data=w, tgt_to_src=(0,0), bcube=obcube, mip=mip)
    # W[(1, 0)] = 0.125
    w = torch.ones((1,1,4,4)) * 0.125
    W.write(data=w, tgt_to_src=(1,0), bcube=bcube, mip=mip)
    W.write(data=w, tgt_to_src=(1,0), bcube=obcube, mip=mip)

    dot_product_task = PairwiseDotProductTask(
                                pairwise_fields=F,
                                pairwise_weights=W,
                                output_fields=CF.layers[0],
                                mip=mip,
                                pad=0,
                                crop=0,
                                bcube=bcube)
    dot_product_task.execute()

    cf = torch.zeros((1,2,4,4)) 
    cf[:,0,:,:] = 0
    cf[:,1,:,:] = 1.25
    pf = CF.read((0,0), bcube=bcube, mip=mip)
    assert(torch.equal(pf, cf))

    # F[(0,0)] = CF[(0,0)]: make output a previous field
    F.layers[0] = CF.layers[0]
    dot_product_task.execute()

    cf = torch.zeros((1,2,4,4)) 
    cf[:,0,:,:] = 0
    cf[:,1,:,:] = 1.875
    pf = CF.read((0,0), bcube=bcube, mip=mip)
    assert(torch.equal(pf, cf))

    delete_layer('/tmp/cloudvolume/empty_fields')
    delete_layer('/tmp/cloudvolume/empty_weights')
    delete_layer('/tmp/cloudvolume/fields')
    delete_layer('/tmp/cloudvolume/weights')
    delete_layer('/tmp/cloudvolume/convolved_fields')