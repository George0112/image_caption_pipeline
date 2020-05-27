class mymodel:
    def __init__(self):
        pass

    def predict(self, X, features_names, **kwargs):
        print(X)
        print(kwargs)
        return ["hello", "world"]
