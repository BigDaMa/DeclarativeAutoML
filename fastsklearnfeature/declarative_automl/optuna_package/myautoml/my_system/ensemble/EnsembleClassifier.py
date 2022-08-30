import numpy as np

class EnsembleClassifier:
    def __init__(self, models, ensemble_selection):
        self.models = models
        self.ensemble_selection = ensemble_selection

    def predict(self, X_test):
        if len(self.models) == 1:
            return self.models[0].predict(X_test)
        else:
            test_predictions = []
            for m in self.models:
                test_predictions.append(m.predict_proba(X_test))
            y_hat_test_temp = self.ensemble_selection.predict(np.array(test_predictions))
            y_hat_test_temp = np.argmax(y_hat_test_temp, axis=1)
            return y_hat_test_temp