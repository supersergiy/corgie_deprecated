from cv_types import CV_TYPE_LIST, ImgCV, MaskCV, FieldCV
from jobs import JobWithDependencies, JobWithoutDependencies

class ApplyComputeFieldModelJob(JobWithDependencies):
    def __init__(self, model_name, mode_params, mip_in, mip_out,
                src_sec, tgt_sec, dst_domain,
                img_domains, mask_domains=[],
                chunk_size=None
                ):
        self.chunk_size   = chunk_size
        self.model_name   = model_name
        self.model_params = model_params
        self.mip_in       = mip_in
        self.mip_out      = mip_out
        self.domains      = {
                    ImgCV: img_domains,
                    MaskCV: mask_domains
                }
        self.src_sec = src_sec
        self.tgt_sec = tgt_sec
        self.dst_domain = dst_domain

    def __call__(self, src_sec, tgt_sec, dst_domain):
        prepare_data_job_list = self.get_prepare_data_job_list()
        yield prepare_data_job_list

        # 1. clean up src_sec and tgt_sec from unneeded fields :
        #        leave the specified images and masks, and all fields

        # 2. break the src_sec and tgt_sec into patches
        #    get a patch list

        # 3. create a list of PatchProcessing Tasks
        #    return it


    def get_prepare_data_job_list(self):
        # we assume that these downsampling tasks have
        # no dependencies between them, and that each is
        # a JobWithoutDependencies
        job_list = []
        for sec in [self.src_sec, self.tgt_sec]:
            for cv_type in [ImgCV, MaskCV, FieldCV]:
                if cv_type == FieldCV:
                    domain_list = sec.get_domains_of_type(FieldCV)
                else:
                    domain_list = self.domains[cv_type]

                # TODO: have custom, non-default downsamplers per domain
                downsample_job_constructor = cv_type.get_downsample_job_constructor

                for domain in domain_list:
                    downsample_tasks = None
                    mips_with_data = sec.domains[cv_type][domain].mips_with_data

                    if cv_type == ImgCV:
                        # has to have data at this MIP
                        if self.in_mip not in mips_with_img_data:
                            # has to be downsampled from lower MIP
                            assert self.in_mip > mips_with_img_data.min()
                            start_mip = mips_with_img_data.min()
                            for m in mips_with_img_data:
                                if m > start_mip and  m < self.in_mip:
                                    start_mip = m
                            dowsample_job = downsample_job_constructor(
                                sec[cv_type][domain],
                                start_mip=start_mip,
                                end_mip=self.mip_in)
                            job_list.append(downsample_job)

                    if cv_type in [MaskCV, FieldCV]:
                        # has to have data at this MIP or Above
                        if mips_with_data.max() < self.mip_in:
                            start_mip = mips_with_data.max()
                            dowsample_job = downsample_job_constructor(
                                sec[cv_type][domain],
                                start_mip=start_mip,
                                end_mip=self.mip_in)
                            job_list.append(downsample_job)
        return job_list


def ComputeFieldPatchTask:
    def __init__(self):
        # save all params, including src and
        pass

    def __call__(self):
        # load model and everything
        pass

class ComputeFieldJob():
    def __int__(self):
        pass

    def prepare_data(self, src_sec, dst_sec, src_field_domain, dst_image_domain):
        pass

    def get_task_gen(self):
        raise NotImplementedError


class MultistageComputeFieldMethod(MethodBase):
    def __init__(self, stages):
        self.stages = stages

    def get_needed_mips(self):
        result = []
        for s in stages:
            result.extend(s.get_needed_mips())
        return result

    def get_task_gen(self, src_sec, dst_sec, src_field_domain, dst_image_domain):
        def multistage_task_gen():
            for s in self.stages:
                task_gen = s.get_task_gen()
                for tasks in task_gen:
                    yield tasks

                # if it is that simple, it should be a general "Sequence" task
                # Stage 1 needs images at the right MIP, pre-existing field
                #         at the right MIP
                # Stage 2 needs images + the field of Stage 1 at Stage 2 MIP
                # Stage 3 needs images + the field of both stage 1 and stage 2 at MIP

                # should i combine stage1 and 2 fields or leave them separate?

                # maybe re-compute encodings -- ncc? <- l8r











