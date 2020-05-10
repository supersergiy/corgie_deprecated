import copy

from corgie import scheduling
from corgie.fields  import img_to_field_info
from corgie.stack   import Stack, FieldDomain
from corgie.cli.copy    import get_copy_stack_tasks

def align_block(scheduler, input_stack, output_stack, align_method,
        render_method, copy_start, backwards, write_info):

    align_block_task_gen = AlignBlockJob(input_stack,
                                         output_stack,
                                         compute_field_method,
                                         render_method,
                                         copy_start,
                                         backward,
                                         write_info)
    scheduler.register_job(job_name='block_alignment',
                           job_task_generator=align_block_task_gen)
    scheduler.execute_until_completion()

def align_block_bidirectional(scheduler, input_stack, output_stack, align_method,
        render_method, copy_start, backwards, write_info):
    raise NotImplementedError

class AlignBlockJob(scheduling.Job):
    def __init__(input_stack,
                 output_stack
                 compute_field_method,
                 render_method,
                 copy_start,
                 backward=False,
                 write_info=False):
        if backward:
            self.z_start = input_stack.z_range()[1]
            self.z_end = input_stack.z_range()[0]
            self.z_step = 1
        else:
            self.z_start = input_stack.z_range()[0]
            self.z_end = input_stack.z_range()[1]
            self.z_step = -1

        self.input_stack = input_stack
        self.output_stack = output_stack
        self.compute_field_method = compute_field_method
        self.render_method = render_method
        self.copy_start = copy_start
        self.backward = backward
        self.write_info = write_info

    def __execute__():
        self.output_stack.create_domain('block_aligned_img',
                info=self.input_stack[ImgCV]['img'].info,
                cv_type=ImgCV,
                write_info=self.write_info)

        self.input_stack.create_domain('aignment_field',
                domain_type=FieldDomain,
                info=img_to_field_info(self.input_stack[ImgCV]['img'].info),
                cv_type=FieldCV,
                write_info=self.write_info)


        if self.copy_start:
            start_sec = self.input_stack.z_cutout(z_start, z_start + 1)
            copy_job = CopyJob(input_stack=start_sec,
                               dst_cv=dst_cv,
                               needed_mips=needed_mips,
                               offset=None)
            yield copy_job
            yield scheduling.wait_until_done

        this_aligned_sec = start_sec

        for z in range(self.z_start, self.z_end + self.z_step, self.z_step):
            curr_sec = self.input_stack.z_cutout(z, z + 1)
            prev_alinged_sec = self.this_aligned_sec
            this_alinged_sec = self.output_stack.z_cutout(z, z + 1)

            compute_field_job = compute_field_method(
                    src_sec=curr_sec,
                    tgt_sec=prev_aligned_sec,
                    dst_domain='alignment_field')

            yield compute_field_job
            yield scheduling.wait_until_done

            render_job = render_method(
                    src_sec=curr_sec,
                    dst_sec=this_aligned_sec,
                    src_field_domain='alignment_field',
                    dst_image_domain='block_aligned_img'
                    )

            yield render_job
            yield scheduling.wait_until_done


