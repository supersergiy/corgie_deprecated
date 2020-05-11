class CorgieException(Exception):
    pass


class IncorrectArgumentDefinition(CorgieException):
    def __init__(self, command, argument, argtype, reason=''):
        self.msg = "\nCommand: {}, Argument: {} {}\n Reason: {}".format(command,
                argtype, argument, reason)
        super().__init__(self.msg)


class ArgumentError(CorgieException):
    def __init__(self, argument, reason=''):
        self.msg = "\nArgument: {}\n Reason: {}".format(argument, reason)
        super().__init__(self.msg)


class NoMipDataException(CorgieException):
    def __init__(self, layer, mip):
        self.msg = "\nLayer: {}\n MIP: {} declared to have no data."\
            " Refer to TODO for more info".format(str(layer), mip)
        super().__init__(self.msg)


class ReadError(CorgieException):
    def __init__(self, layer, reason):
        self.msg = "\nLayer '{}': \n{}".format(str(layer), reason)
        super().__init__(self.msg)


class WriteError(CorgieException):
    def __init__(self, layer, reason):
        self.msg = "\nLayer '{}': \n{}".format(str(layer), reason)
        super().__init__(self.msg)
