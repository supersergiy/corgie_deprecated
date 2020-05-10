def sergiy():
    print ("hey")
    for x in range(0, 10):
        print ("hi")
        yield x

ira = sergiy()
import pdb; pdb.set_trace()
