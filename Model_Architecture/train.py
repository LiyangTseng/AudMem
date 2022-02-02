import warnings
warnings.filterwarnings("ignore")
import yaml
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("--model", help="h_lstm, h_mlp, e_crnn, e_pase, e_pasep, pase_mlp", default="e_pasep")
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
    parser.add_argument('--features', default="all",
                    help='chords/rhythm/timbre', required=False)


    paras = parser.parse_args()
    setattr(paras, 'gpu', not paras.cpu)
    setattr(paras, 'verbose', not paras.no_msg)

    config = yaml.load(open("config/{}.yaml".format(paras.model), 'r'), Loader=yaml.FullLoader)
 
    if paras.lr_rate != None:
        config["hparas"]["optimizer"]["lr"] = paras.lr_rate

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
        from bin.train_h_lstm import Solver
    elif paras.model == "h_mlp":
        from bin.train_h_mlp import Solver
    elif paras.model == "e_crnn":
        from bin.train_e_crnn import Solver
    elif paras.model == "e_pase":
        from bin.train_e_pase import Solver
    elif paras.model == "e_pasep":
        from bin.train_e_pasep import Solver
    elif paras.model == "pase_mlp":
        from bin.train_pase_mlp import Solver        
    else:
        raise Exception("Not Implement Error")

    
    solver = Solver(config=config, paras=paras, mode="train")
    # TODO: include k-fold
    solver.load_data()
    solver.set_model()
    solver.exec()

    
 