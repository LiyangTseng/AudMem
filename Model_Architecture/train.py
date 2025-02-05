import warnings
warnings.filterwarnings("ignore")
import yaml
import argparse

if __name__ == "__main__":
    
    models = ["h_lstm", "h_mlp", "h_svr", "e_crnn", "e_cnn", "e_transformer", "e_pase", "e_pasep", "pase_mlp", "pase_lstm", "probing", "ssast"]

    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("--model", help=",".join(models), default="h_lstm")
    parser.add_argument("--patience", default=10, type=int, help="early stop patience")
    parser.add_argument('--name', default=None, type=str, help='Name for logging.')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--logdir', default='tensorboard/', type=str,
                    help='Logging path.', required=False)
    parser.add_argument('--ckpdir', default='weights/', type=str,
                    help='Checkpoint path.', required=False)
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
    parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
    parser.add_argument('--lr_rate', default=None,
                    help='customized learning rate', required=False)
    parser.add_argument('--svr_kernel', default=None,
                    help='customized svr kernel', required=False)
    parser.add_argument('--features', default="all",
                    help='all/harmony/rhythm/timbre/harmony-rhythm/harmony-timbre/rhythm-timbre', required=False)  
    parser.add_argument('--no_kfold', action='store_true',
                    help='do k-fold validation or not')  
    parser.add_argument('--kfold_splits', default=10, type=int,
                    help='number of k-fold splits', required=False)
    parser.add_argument('--fold_index', default=9, type=int,
                    help='index of 10 fold', required=False)
    parser.add_argument('--seed', default=1234, type=int,
                    help='random seed', required=False)
    parser.add_argument('--use_pitch_shift', action='store_true',
                    help='use pitch shifting as data augmentation or not')
    parser.add_argument('--use_lds', default=None,
                    help='use label distribution smoothing or not', required=False)
    


    paras = parser.parse_args()
    setattr(paras, 'gpu', not paras.cpu)
    # setattr(paras, 'gpu', False)
    setattr(paras, 'verbose', not paras.no_msg)

    config = yaml.load(open("config/{}.yaml".format(paras.model), 'r'), Loader=yaml.FullLoader)
 
    if paras.lr_rate != None:
        config["hparas"]["optimizer"]["lr"] = paras.lr_rate
    if paras.svr_kernel != None:
        config["model"]["kernel"] = paras.svr_kernel

    if paras.use_lds != None:
        config["model"]["use_lds"] = paras.use_lds == "True"

    if paras.features == "all":
        pass
    elif paras.features == "harmony":
        config["features"] = {'chords': ['chroma', 'tonnetz'], 'emotions': ['static_arousal', 'static_valence'] }
        config["model"]["sequential_input_size"] = 18
    elif paras.features == "timbre":
        config["features"] = {'timbre': ["mfcc", "mfcc_delta", "mfcc_delta2"], 'emotions': ['static_arousal', 'static_valence']}
        config["model"]["sequential_input_size"] = 60
    elif paras.features == "rhythm":
        config["features"] = {'rhythms': ["tempogram"], 'emotions': ['static_arousal', 'static_valence']}
        config["model"]["sequential_input_size"] = 384
    elif paras.features == "harmony-timbre":
        config["features"] = {'chords': ['chroma', 'tonnetz'], 'timbre': ["mfcc", "mfcc_delta", "mfcc_delta2"], 'emotions': ['static_arousal', 'static_valence']}
        config["model"]["sequential_input_size"] = 78
    elif paras.features == "harmony-rhythm":
        config["features"] = {'chords': ['chroma', 'tonnetz'], 'rhythms': ["tempogram"], 'emotions': ['static_arousal', 'static_valence']}
        config["model"]["sequential_input_size"] = 402
    elif paras.features == "timbre-rhythm":
        config["features"] = {'timbre': ["mfcc", "mfcc_delta", "mfcc_delta2"], 'rhythms': ["tempogram"], 'emotions': ['static_arousal', 'static_valence']}
        config["model"]["sequential_input_size"] = 442
    else:
        raise Exception("Not Implement Error")

    # ref: https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    if paras.model in models:
        Solver = getattr(__import__("bin.train_{}".format(paras.model), fromlist=["Solver"]), "Solver")
    else:
        raise Exception("Not Implement Error")

    
    solver = Solver(config=config, paras=paras, mode="train")
    solver.load_data()
    solver.set_model()
    solver.exec()

    
 