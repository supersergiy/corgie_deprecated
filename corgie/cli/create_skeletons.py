import click
from copy import deepcopy

from corgie import scheduling, argparsers, helpers, stack

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords

from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec

from corgie.cli.render import RenderJob
from corgie.cli.copy import CopyJob
from corgie.cli.compute_field import ComputeFieldJob

import kimimaro


class SkeletonJob(scheduling.Job):
    def __init__(
        self, seg_layer, dst_path, bcube, chunk_xy, chunk_z, 
        mip, teasar_params, 
        object_ids=None,
        fix_branching=True, fix_borders=True, fix_avocados=False,
        dust_threshold=0, tick_threshold=1000
    ):
        self.seg_layer = seg_layer
        self.dst_path = dst_path
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mip = mip
        self.teasar_params = teasar_params
        self.object_ids = object_ids
        self.fix_branching = fix_branching
        self.fix_borders = fix_borders
        self.fix_avocados = fix_avocados
        self.dust_threshold = dust_threshold
        self.tick_threshold = tick_threshold
        super().__init__()

    def task_generator(self):
        chunks = self.seg_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=mip)
        tasks = [SkeletonTask(self.seg_layer,
                            self.dst_path,
                            mip=mip,
                            teasar_params=self.teasar_params,
                            object_ids=self.object_ids,
                            dust_threshold=self.dust_threshold,
                            fix_branching=self.fix_branching,
                            fix_borders=self.fix_borders,
                            fix_avocados=self.fix_avocados,
                            bcube=input_chunk)
                            for input_chunk in chunks]
        corgie_logger.info(f"Yielding render tasks for bcube: {self.bcube}, MIP: {mip}")
        yield tasks
        # yield scheduling.wait_until_done

class SkeletonTask(scheduling.Task):
    def __init__(self, seg_layer, dst_path, bcube, 
        mip, teasar_params, 
        object_ids,
        fix_branching, fix_borders, fix_avocados,
        dust_threshold, tick_threshold):
        super().__init__(self)
        self.seg_layer = seg_layer
        self.dst_path = dst_path
        self.bcube = bcube
        self.mip = mip
        self.teasar_params = teasar_params
        self.object_ids = object_ids
        self.fix_branching = fix_branching
        self.fix_borders = fix_borders
        self.fix_avocados = fix_avocados
        self.dust_threshold = dust_threshold

    def execute(self):
        corgie_logger.info(f"Skeletonizing {self.seg_layer} at MIP{self.mip}, region: {self.bcube}")
        seg_data = self.seg_layer.read(bcube=self.bcube, mip=self.mip)
        resolution = self.seg_layer.cv[self.mip].resolution
        skeletons = kimimaro.skeletonize(
            seg_data, self.teasar_params, 
            object_ids=self.object_ids, 
            anisotropy=resolution,
            dust_threshold=self.dust_threshold, 
            progress=self.progress, 
            fix_branching=self.fix_branching,
            fix_borders=self.fix_borders,
            fix_avocados=self.fix_avocados,
        )

        minpt = np.array(self.bcube.x_range(self.mip)[0], self.bcube.y_range(self.mip)[0], self.bcube.z_range

        for segid, skel in skeletons:
            skel.vertices[:] += self.bcube.minpt(self.mip)

        skeletons = skeletons.values()

        with Storage(path, progress=vol.progress) as stor:
            for skel in skeletons:
                stor.put_file(
                    file_path="{}:{}".format(skel.id, bcube.to_filename(self.mip)),
                    content=pickle.dumps(skel),
                    compress='gzip',
                    content_type="application/python-pickle",
                    cache_control=False,
                )


# @corgie_option('--dst_layer_spec', '-t', nargs=1,
#         type=str, required=True, multiple=True,
#         help='Target layer spec. Use multiple times to include all masks, fields, images. \n'\
#                 'DEFAULT: Same as source layers')
# @corgie_option('--vector_field_layer_spec',  '-s', nargs=1,
#         type=str, required=True, multiple=True,
#         help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
#                 LAYER_HELP_STR)
# @corgie_option('--dst_unaligned_skel_layer_spec', '-t', nargs=1,
#         type=str, required=False, multiple=True,
#         help='Target layer spec. Use multiple times to include all masks, fields, images. \n'\
#                 'DEFAULT: Same as source layers')


@click.command()
# Layers
@corgie_optgroup('Layer Parameters')
@corgie_option('--seg_layer_spec',  '-s', nargs=1,
        type=str, required=True,
        help='Seg layer from which to skeletonize segments.')
@corgie_option('--dst_path',  nargs=1, type=str, required=True,
        help="Folder where to store the skeletons")

@corgie_optgroup('Skeletonization Parameters')
@corgie_option('--mip', nargs=1, type=int, default=2)
@corgie_option('--teasar_scale', nargs=1, type=int, default=10)
@corgie_option('--teasar_const', nargs=1, type=int, default=10)
@corgie_option('--ids', multiple=True, type=int, help='Segmentation ids to skeletonize')
@corgie_option('--ids_filepath', type=str, help='File containing segmentation ids to skeletonize')
@corgie_option('--tick_threshold', nargs=1, type=int, default=1000)
@corgie_option('--chunk_xy', nargs=1, type=int, default=256)
@corgie_option('--chunk_z', nargs=1, type=int, default=512)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',        nargs=1, type=str, required=True)
@corgie_option('--end_coord',          nargs=1, type=str, required=True)
@corgie_option('--coord_mip',          nargs=1, type=int, default=0)

@click.pass_context
def create_skeletons(ctx, seg_layer_spec, dst_folder, mip, teasar_scale, teasar_const,
        ids, ids_filepath, tick_threshold, chunk_xy, chunk_z, start_coord, end_coord, coord_mip):
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    seg_stack = create_stack_from_spec(seg_layer_spec,
            name='src', readonly=True)
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    object_ids = ids
    if ids_filepath is not None:
        object_ids = []
        with open(ids_filepath, 'r') as f:
            line = f.readline()
            while line:
                object_ids.append(int(line))
                line = f.readline()
    if object_ids is None or len(object_ids) == 0:
      raise ValueError('Must specify ids to skeletonize')
    teasar_params = {'scale': teasar_scale, 'const': teasar_const}
    
    
    seg_layer = src_stack.get_layers_of_type("segmentation")[0]
    skeleton_job = SkeletonJob(seg_layer=seg_layer,
                                dst_path=dst_path,
                                bcube=bcube,
                                chunk_xy=chunk_xy,
                                chunk_z=chunk_z,
                                mip=mip,
                                teasar_params=teasar_params,
                                object_ids=object_ids,
                                tick_threshold=tick_threshold)

    scheduler.register_job(skeleton_job, job_name="Skeltonize {}".format(bcube))

    scheduler.execute_until_completion()
    result_report = f"Skeletonized {str(seg_layer)}. "
    corgie_logger.info(result_report)



