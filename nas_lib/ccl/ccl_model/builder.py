from nas_lib.predictor_retrain_compare.predictor_gin_ccl import PredictorGINCCL


def build_model(model_type, with_g_func=True):
    if model_type == 'SS_CCL':
        return PredictorGINCCL
    else:
        raise NotImplementedError(f'The model type {model_type} has not implemented!')