import shap
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency, DeepLift

class ExplainabilityModule:
    def __init__(self, model):
        self.model = model

    def explain(self, input_tensor, method='integrated_gradients'):
        if method == 'integrated_gradients':
            ig = IntegratedGradients(self.model)
            attributions = ig.attribute(input_tensor)
        elif method == 'saliency':
            saliency = Saliency(self.model)
            attributions = saliency.attribute(input_tensor)
        elif method == 'deep_lift':
            dl = DeepLift(self.model)
            attributions = dl.attribute(input_tensor)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        return attributions