

# TODO: use macropy3
# maybe instead of macros i should yield full jobs.
# at the end, the guy who needs to break it into tasks
# will recursively pop them. if it's a job, keep milking
# if it's a task list, accept it
# what if it's a list of jobs?
# possible
# assume it can only be a list of simple jobs without dependencies
# then just pop them all

def get_yield_until_emtpy_macro(generator_name):
    return \
'try:\
    tasks = {}\
except StopIteration:
    break
else:
    yield compute_field_tasks
'.format(generator_name)
