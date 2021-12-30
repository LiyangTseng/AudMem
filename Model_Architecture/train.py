import warnings
warnings.filterwarnings("ignore")
import yaml
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument("--model", help="h_lstm, h_mlp", default="h_lstm")
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


    paras = parser.parse_args()
    setattr(paras, 'gpu', not paras.cpu)
    setattr(paras, 'verbose', not paras.no_msg)

    config = yaml.load(open("config/{}.yaml".format(paras.model), 'r'), Loader=yaml.FullLoader)
 
    if paras.model == "h_lstm":
        from bin.train_h_lstm import Solver
    else:
        raise Exception("Not Implement Error")

    solver = Solver(config=config, paras=paras, mode="train")
    solver.load_data()
    solver.set_model()
    solver.exec()

    
 