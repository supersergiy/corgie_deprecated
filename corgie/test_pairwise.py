import pytest
import torch
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec
from pairwise import PairwiseStack

import shutil
import os

from corgie.boundingcube import BoundingCube
from corgie.mipless_cloudvolume import MiplessCloudVolume
from corgie.data_backends import str_to_backend, CVDataBackend

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path) 

def test_pairwisestack():
    mip=0
    sz = Vec(*[4, 4, 1])
    info = CloudVolume.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='float32', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,-1], 
             volume_size=[8,8,3],
             chunk_size=sz,
             )
    ref = MiplessCloudVolume(path='file:///tmp/cloudvolume/empty_volume',
                             info=info,
                             overwrite=True)

    def get_bcube(sz, x_block, y_block, z):
        return BoundingCube(xs=x_block*sz, xe=(x_block+1)*sz, 
                           ys=y_block*sz, ye=(y_block+1)*sz, 
                           zs=z, ze=z+1,
                           mip=mip)

    F = PairwiseStack(name='test')
    backend = CVDataBackend(device='cpu')
    for o in [1,-2]:
        spec = {'name': o,
                'layer_type': 'field',
                'readonly': False,
                'path': 'file:///tmp/cloudvolume/empty_volume/{}'.format(o)}
        layer = backend.create_layer(**spec, reference=ref)
        F.addlayer(layer)
    for o in [1,-2]:
        assert(o in F.layers)

    # F[(1,0)]
    bcube = get_bcube(sz=4, x_block=0, y_block=1, z=0)
    g = torch.zeros((1,2,4,4))
    g[:,0,:,:] = 4
    g[:,1,:,:] = 0
    F.layers[1].write(g, bcube=bcube, mip=mip) 
    assert(torch.equal(F.read((1,0), bcube=bcube, mip=mip), g))

    # F[(-1,1)]
    bcube = get_bcube(sz=4, x_block=0, y_block=0, z=1)
    g = torch.zeros((1,2,4,4))
    g[:,0,:,:] = 0
    g[:,1,:,:] = 4
    F.layers[-2].write(g, bcube=bcube, mip=mip) 
    assert(torch.equal(F.read((-1,1), bcube=bcube, mip=mip), g))

    o = torch.zeros((1,2,4,4))
    o[:,0,:,:] = 4
    o[:,1,:,:] = 4
    x = F.read((-1,1,0), bcube=bcube, mip=mip)
    assert(torch.equal(x,o))

    with pytest.raises(ValueError) as e:
        x = F.read((-1,), bcube=bcube, mip=mip)
    with pytest.raises(ValueError) as e:
        x = F.read((-3,1), bcube=bcube, mip=mip)

    delete_layer('/tmp/cloudvolume/empty_volume')