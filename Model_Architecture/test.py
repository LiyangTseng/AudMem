import warnings
warnings.filterwarnings("ignore")
import yaml
import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='test config')
    parser.add_argument("--model", help="h_lstm, h_mlp, e_crnn", default="h_lstm")
    parser.add_argument('--cpu', action='store_true', help='Disable GPU inferencing.')
    parser.add_argument('--ckpdir', default='weights/', type=str,
                    help='Checkpoint path.', required=False)
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
    parser.add_argument('--outdir', default='result/', type=str,
                    help='Prediction output path.', required=False)
    # parser.add_argument('--load', default="weights/train_memorability/Regression_LSTM/Regression_LSTM.pt", type=str,
    parser.add_argument('--load', default="weights/h_lstm/22-01-10_20:58/h_lstm_6.pth", type=str,
                    help='ckpt path of pre-trained model', required=False)


    paras = parser.parse_args()
    setattr(paras, 'gpu', not paras.cpu)
    setattr(paras, 'verbose', not paras.no_msg)

    config = yaml.load(open("config/{}.yaml".format(paras.model), 'r'), Loader=yaml.FullLoader)
 
    if paras.model == "h_lstm":
        from bin.test_h_lstm import Solver
    elif paras.model == "h_mlp":
        from bin.test_h_mlp import Solver
    elif paras.model == "e_crnn":
        from bin.test_e_crnn import Solver
    else:
        raise Exception("Not Implement Error")

    solver = Solver(config=config, paras=paras, mode="test")
    solver.load_data()
    solver.set_model()
    solver.exec()

    
 