

def MultistageComputeFieldFactory(compute_field_stages):
    def __init__(self, stages):
        self.stages = stages

    def get_needed_mips(self):
        result = []
        for s in stages:
            result.extend(s.get_needed_mips())
        return result

    def __call__(self, src_sec, dst_sec, dst_field_domain):
        class MultistageComputeFieldJob(scheduling.Job):
            def __init__(self):
                self.src_sec = src_sec
                self.dst_sec = dst_sec
                self.dst_field_domain = dst_field_domain
                self.stages =

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
        return MoltistageComputeFieldJob



