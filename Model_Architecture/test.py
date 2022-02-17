import warnings
warnings.filterwarnings("ignore")
import yaml
import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='test config')
    parser.add_argument("--model", help="h_lstm, h_mlp, e_crnn, pase_mlp", default="pase_mlp")
    parser.add_argument('--cpu', action='store_true', help='Disable GPU inferencing.')
    parser.add_argument('--ckpdir', default='weights/', type=str,
                    help='Checkpoint path.', required=False)
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
    parser.add_argument('--outdir', default='result/', type=str,
                    help='Prediction output path.', required=False)
    parser.add_argument('--load', default="weights/pase_mlp/fold_0/pase_mlp_best.pth", type=str,
                    help='ckpt path of pre-trained model', required=False)
    parser.add_argument('--features', default="all",
                    help='all/chords/chords-timbre', required=False)
    parser.add_argument('--do_kfold', default=True,
                    help='do k-fold validation or not', required=False)  
    parser.add_argument('--kfold_splits', default=10, type=int,
                    help='number of k-fold splits', required=False)
    parser.add_argument('--fold_index', default=0, type=int,
                    help='index of 10 fold', required=False)

    paras = parser.parse_args()
    setattr(paras, 'gpu', not paras.cpu)
    setattr(paras, 'verbose', not paras.no_msg)

    config = yaml.load(open("config/{}.yaml".format(paras.model), 'r'), Loader=yaml.FullLoader)
 
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

    if paras.model == "h_lstm":
        from bin.test_h_lstm import Solver
    elif paras.model == "h_mlp":
        from bin.test_h_mlp import Solver
    elif paras.model == "e_crnn":
        from bin.test_e_crnn import Solver
    elif paras.model == "pase_mlp":
        from bin.test_pase_mlp import Solver
    else:
        raise Exception("Not Implement Error")

    solver = Solver(config=config, paras=paras, mode="test")
    solver.load_data()
    solver.set_model()
    solver.exec()

    
 