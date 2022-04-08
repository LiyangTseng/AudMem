import warnings
warnings.filterwarnings("ignore")
import yaml
import argparse
if __name__ == "__main__":
    
    models = ["h_lstm", "h_mlp", "h_svr", "e_crnn", "e_cnn", "pase_mlp", "pase_lstm", "e_transformer", "ssast", "random_guess", "mean"]

    parser = argparse.ArgumentParser(description='test config')
    parser.add_argument("--model", help=",".join(models), default="h_svr")
    parser.add_argument('--cpu', action='store_true', help='Disable GPU inferencing.')
    parser.add_argument('--ckpdir', default='weights/', type=str,
                    help='Checkpoint path.', required=False)
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
    parser.add_argument('--outdir', default='default_output', type=str,
                    help='Prediction output path.', required=False)
    parser.add_argument('--load', default="weights/e_cnn/fold_0/e_cnn_best.pth", type=str,
                    help='ckpt path of pre-trained model', required=False)
    parser.add_argument('--features', default="all",
                    help='all/chords/chords-timbre', required=False)
    parser.add_argument('--no_kfold', action='store_true',
                    help='do k-fold validation or not')  
    parser.add_argument('--kfold_splits', default=10, type=int,
                    help='number of k-fold splits', required=False)
    parser.add_argument('--fold_index', default=0, type=int,
                    help='index of 10 fold', required=False)

    paras = parser.parse_args()
    if paras.outdir == "default_output" and paras.no_kfold == False:
        paras.outdir = "results/{}/fold_{}".format(paras.model, paras.fold_index)
    elif paras.outdir == "default_output" and paras.no_kfold == True:
        print("do not use k-fold validation")
        paras.outdir = "results/{}".format(paras.model)
    setattr(paras, 'gpu', not paras.cpu)
    setattr(paras, 'verbose', not paras.no_msg)

    if paras.model not in ["random_guess", "mean"]:
        config = yaml.load(open("config/{}.yaml".format(paras.model), 'r'), Loader=yaml.FullLoader)
    else:
        config = None
        
    if paras.features == "all":
        pass
    elif paras.features == "chords-timbre":
        config["features"] = {'chords': ['chroma'], 'timbre': ['mfcc'], 'emotions': ['static_arousal', 'static_valence']}
        config["model"]["sequential_input_size"] = 32
    elif paras.features == "chords":
        config["features"] = {'chords': ['chroma'], 'emotions': ['static_arousal', 'static_valence']}
        config["model"]["sequential_input_size"] = 12
    else:
        raise Exception("Not Implement Error")

    # ref: https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
    if paras.model in models:
        Solver = getattr(__import__("bin.test_{}".format(paras.model), fromlist=["Solver"]), "Solver")
    else:
        raise Exception("Not Implement Error")
    solver = Solver(config=config, paras=paras, mode="test")
    solver.load_data()
    solver.set_model()
    solver.exec()

    
 