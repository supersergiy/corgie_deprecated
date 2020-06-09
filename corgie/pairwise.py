import torchfields
import numpy as np
from stack import StackBase

class PairwiseTensors(StackBase):
    """Manage set of layers that contain objects specified by pair of neighbors
    
    Object examples include pairwise fields and images related to pairwise fields, 
    such as confidence maps.
    
    Note:
        layers are named with ints that indicate OFFSET, such that
            f_{z+offset \leftarrow z} is stored in OFFSET[Z]
        layers in PairwiseTensor must all be of the same type
    """
    def __init__(self, dtype='img', **kwargs):
        self.dtype=dtype
        super().__init__(**kwargs)

    def addlayer(self, layer):
        """Only allow layers of dtype with ints as names to represent offsets
        """
        assert(isinstance(layer.name, int))
        assert(layer.get_layer_type() == self.dtype)
        super().add_layer(layer)
    
    @property
    def offsets(self):
        return list(self.layers.keys())

    def write(self, data, tgt_to_src, bcube, mip):
        """Save pairwise object at layers[OFFSET][:,:,z]

        Args:
            data: tensor to be written
            tgt_to_src: 2-element iterable TGT & SRC
            bcube: BoundingCube
            mip: int for MIP level 
        """
        if len(tgt_to_src) != 2:
            raise ValueError('len(tgt_to_src) is {} != 2. '
                             'Pairwise objects are only defined between '
                             'a pair of sections.'.format(len(tgt_to_src)))
        tgt, src = tgt_to_src
        offset = tgt - src
        layer = self.layers[offset]
        layer.write(data, bcube=bcube, mip=mip) 

    def read(self, tgt_to_src, bcube, mip):
        """Read pairwise object at layers[OFFSET][:,:,z]

        Args:
            tgt_to_src: 2-element iterable TGT & SRC
            bcube: BoundingCube
            mip: int for MIP level 

        Returns:
            tensor
        """
        if len(tgt_to_src) != 2:
            raise ValueError('len(tgt_to_src) is {} != 2. '
                             'Pairwise objects are only defined between '
                             'a pair of sections.'.format(len(tgt_to_src)))
        tgt, src = tgt_to_src
        offset = tgt - src
        layer = self.layers[offset]
        return layer.read(bcube=bcube, mip=mip) 

class PairwiseFields(PairwiseTensors):
    """Manage set of layers that contain fields between neighbors
    
    We use the following notation to define a pairwise field which aligns
    section z to section z+k.
    
        $f_{z+k \leftarrow z}$
    
    It's easiest to interpret this as the displacement field which warps
    section z to look like section z+k. Because of this interpretation,
    section z is considered the SOURCE and z+k is the TARGET, or

        $f_{TARGET \leftarrow SOURCE}
    
    We define the offset as the distance between the SOURCE and the TARGET. So
    in the case above, the offset is k.
    
    We store this field in a layer where the path indicates the offset,
    and the actual field will be stored at cv[..., z].
    
    One purpose of this class is to easily compose pairwise fields together.
    For example, if we wanted to create the field:
    
        $f_{z+k \leftarrow z+j} \circ f_{z+j \leftarrow z}$ 
    
    Then we can access it with the convention:
    
        ```
        F = PairwiseFields(path, offsets, bbox, mip)
        f = F[(z+k, z+j, z)]
        ```
    
    Note:
        path: directory with pairwise field format
            ./{OFFSETS}
            $f_{z+offset \leftarrow z}$ is stored in OFFSET[Z]
        offsets: list of ints indicating offset (the distance from source to
            targets).
    """
    def __init__(self, **kwargs):
        super().__init__(dtype='field', **kwargs)

    def read(self, tgt_to_src, bcube, mip):
        """Get field created by composing fields accessed by z_list[::-1]

        Args:
            tgt_to_src: list of ints, sorted from target to source, e.g.
                $f_{0 \leftarrow 2} \circ f_{2 \leftarrow 3}$ : [0, 2, 3]
            bcube: BoundingCube
            mip: int for MIP level

        Returns:
            torchfields.DisplacementField
        """
        if len(tgt_to_src) < 2:
            raise ValueError('len(tgt_to_src) is {} != 2. '
                     'Pairwise objects are only defined between '
                     'a pair of sections.'.format(len(tgt_to_src)))
        offsets = np.array([t-s for t,s in zip(tgt_to_src[:-1], tgt_to_src[1:])])
        unavailable = any([o not in self.layers for o in offsets])
        if unavailable:
            raise ValueError('Requested offsets {} are '
                             'unavailable'.format(offsets[unavailable]))
        # tgt_to_src[0] (the ultimate target) is only needed to compute initial offset
        z = tgt_to_src[1]
        offset = offsets[0]
        layer = self.layers[offset]
        obcube = bcube.reset_coords(zs=z, ze=z+1, in_place=False)
        agg_field = layer.read(bcube=obcube, mip=mip).field_()
        for z, offset in zip(tgt_to_src[2:], offsets[1:]):
            trans = agg_field.mean_finite_vector(keepdim=True)
            trans = (trans // (2**mip)) * 2**mip
            layer = self.layers[offset]
            obcube = bcube.reset_coords(zs=z, ze=z+1, in_place=False)
            obcube = obcube.translate(x=int(trans[0,0,0,0]), 
                                      y=int(trans[0,1,0,0]))
            agg_field -= trans
            this_field = layer.read(bcube=obcube, mip=mip).field_()
            agg_field = agg_field(this_field)
            agg_field += trans
        return agg_field
