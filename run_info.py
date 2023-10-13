class run_info:
    def __init__(self, model_class_name: str, model: object):
        self.model = model
        self.model_class_name = model_class_name

    def get_run_name(self):
        return self.model_class_name
