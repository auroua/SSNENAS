import torch
from nas_lib.predictors.predictor_unsupervised_siamese_ged import PredictorSiameseGED


def load_modify_model(model, model_temp, model_path, verbose=0):
    model_state_dict = model.state_dict()
    model_temp.load_state_dict(torch.load(model_path))
    model_temp_state_dict = model_temp.state_dict()
    for name, param in model_temp_state_dict.items():
        if name in model_state_dict:
            if verbose:
                print(f'Model parameter {name} have loaded!')
            model_state_dict[name] = param
    model.load_state_dict(model_state_dict)
    return model


def get_temp_model(predictor_type, input_dim):
    if predictor_type == 'ss_rl':
        predictor = PredictorSiameseGED(input_dim=input_dim)
    else:
        raise NotImplementedError(f'The predictor type {predictor_type} have implement yet!')
    return predictor


def load_predictor_ged_moco(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            # remove prefix
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        elif k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model


def load_predictor_ged_moco_v2(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and 'fc' not in k:
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        elif k.startswith('module.encoder_q') and 'fc' not in k:
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model


def load_supervised_gin_model(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    return model