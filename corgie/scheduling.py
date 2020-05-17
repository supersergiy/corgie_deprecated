import mazepa

wait_until_done = mazepa.Barrier
Task = mazepa.Task
Executor = mazepa.Executor

class Job(mazepa.Job):
    def __init__(self, *kargs, **kwargs):
        self.task_generator = self.task_generator()

    def task_generator():
        raise NotImplemented("Job classes must implement "
                "'task_generator' function")

    def get_tasks(self):
        return next(self.task_generator)

class Scheduler(mazepa.Scheduler):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

def create_scheduler(*kargs, **kwargs):
    return Scheduler(*kargs, **kwargs)

