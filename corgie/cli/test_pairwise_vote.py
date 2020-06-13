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
from corgie.cli.pairwise_vote import PairwiseVoteTask

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_pairwise_vote_task():
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
    offsets = [1,2,3]
    F  = PairwiseFields(name='estimated_fields',
                        folder='file:///tmp/cloudvolume/estimated_fields')
    F.add_offsets(offsets, readonly=False, reference=field_ref)
    for o in offsets: 
        assert(o in F.layers)
    CF = PairwiseFields(name='corrected_fields',
                        folder='file:///tmp/cloudvolume/corrected_fields')
    CF.add_offsets(offsets, readonly=False, reference=F)
    CW = PairwiseTensors(name='corrected_weights',
                        folder='file:///tmp/cloudvolume/corrected_weights')
    CW.add_offsets(offsets, dtype='float32', readonly=False, reference=weight_ref)

    # Want vector voting to be on three vectors that are each rotated by 2*\pi / 3
    # Translate so that average is (0, 1)

    # F[(3, 0)] = (0,4)
    f = torch.zeros((1,2,4,4))
    f[:,0,:,:] = 0
    f[:,1,:,:] = 4
    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=0)
    F.write(data=f, tgt_to_src=(3,0), bcube=bcube, mip=mip)
    # F[(3, 1)] = (0,4)
    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=1)
    F.write(data=f, tgt_to_src=(3,1), bcube=bcube, mip=mip)
    # F[(3, 2)] = (0,4)
    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=2)
    F.write(data=f, tgt_to_src=(3,2), bcube=bcube, mip=mip)
    # F[(2, 0), new_bbox] = (2*sqrt(3), -2-3)
    # F[(3, 2, 0)] = (2*sqrt(3), -2)
    f = torch.zeros((1,2,4,4))
    f[:,0,:,:] = 2*np.sqrt(3)
    f[:,1,:,:] = -5
    bcube = get_bcube(sz=4, x_block=0, y_block=1, z=0)
    F.write(data=f, tgt_to_src=(2,0), bcube=bcube, mip=mip)
    # F[(3, 1), new_bbox] = (-2*sqrt(3), -2-3)
    # F[(3, 1, 0)] = (-2*sqrt(3), -2)
    f = torch.zeros((1,2,4,4))
    f[:,0,:,:] = -2*np.sqrt(3)
    f[:,1,:,:] = -5
    bcube = get_bcube(sz=4, x_block=0, y_block=1, z=0)
    F.write(data=f, tgt_to_src=(1,0), bcube=bcube, mip=mip)

    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=0)
    vote_task = PairwiseVoteTask(estimated_fields=F,
                                corrected_fields=CF,
                                corrected_weights=CW,
                                mip=mip,
                                pad=0,
                                crop=0,
                                bcube=bcube,
                                paths={3: [( 3, 0), ( 3,  2, 0), ( 3,  1, 0)]},
                                softmin_temp=10000, # make all vectors equal
                                blur_sigma=1)
    # We'll vote for F[(1,0)], but the rest will have errors
    # with pytest.raises(ValueError) as e:
    vote_task.execute()

    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=0)
    cx = torch.zeros((1,2,4,4)) 
    cx[:,0,:,:] = 0
    cx[:,1,:,:] = 2/3
    px = CF.read((3,0), bcube=bcube, mip=mip)
    assert(torch.allclose(px, cx, atol=1e-3))
    cw = torch.ones((1, 1, 4, 4))*2
    pw = CW.read((3,0), bcube=bcube, mip=mip)
    assert(torch.allclose(pw, cw))

    delete_layer('/tmp/cloudvolume/empty_fields')
    delete_layer('/tmp/cloudvolume/empty_weights')
    delete_layer('/tmp/cloudvolume/estimated_fields')
    delete_layer('/tmp/cloudvolume/corrected_fields')
    delete_layer('/tmp/cloudvolume/corrected_weights')