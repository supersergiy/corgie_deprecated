import modelhouse

class ApplyModelProcessor():
    def __init__(self, model_path, model_params, **kwargs):
        self.model_path = model_path
        self.model_params = model_params
        self.constructor_kwargs = kwargs

    def run(**kwargs):
        model = modelhouse.load_model(model_path=self.model_path,
                model_params=self.model_params)
        return model(**kwargs, **self.constructor_kwargs)
