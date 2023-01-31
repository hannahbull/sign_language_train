import configargparse

def save_opts(args, fn):
    with open(fn, 'w') as fw:
        for items in vars(args):
            fw.write('%s %s\n' % (items, vars(args)[items]))

def str2bool(v):
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    elif v in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean argument needs to be true or false. '
                        'Instead, it is %s.' % v)

def load_opts():

    parser = configargparse.ArgumentParser(description="main")
    parser.register('type', bool, str2bool)

    parser.add('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--gpu_id', type=str, default='0', help='-1: all, 0-7: GPU index')

    parser.add_argument('--resume', type=str, default=None, help='', nargs='+')
    parser.add_argument('--test_only', action='store_true', help='Run only evaluation')
    parser.add_argument('--n_workers', type=int, default=0, help='Num data workers')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--random_subset_data', type=int, default=1e9, help='Choose random subset of data (for debugging)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Checkpoints directory to save model')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')

    parser.add_argument('--load_vocab', type=bool, default=False, help='Load vocabulary file')
    parser.add_argument('--vocab_file_loc', type=str, default='data/vocab.pkl', help='Vocabulary file')

    parser.add_argument('--train_ids', type=str, default='data/train.txt', help='List of train video ids')
    parser.add_argument('--train_data_loc', type=str, default='/scratch/shared/beegfs/hbull/shared-datasets/mediapi-rgb/videos/features', help='Location of training data')
    parser.add_argument('--train_labels_loc', type=str, default='/scratch/shared/beegfs/hbull/shared-datasets/mediapi-rgb/videos/subtitles.pkl', help='Location of training labels')
    parser.add_argument('--val_ids', type=str, default='data/val.txt', help='List of train video ids')
    parser.add_argument('--val_data_loc', type=str, default='/scratch/shared/beegfs/hbull/shared-datasets/mediapi-rgb/videos/features', help='Location of val data')
    parser.add_argument('--val_labels_loc', type=str, default='/scratch/shared/beegfs/hbull/shared-datasets/mediapi-rgb/videos/subtitles.pkl', help='Location of val labels')
    parser.add_argument('--test_ids', type=str, default='data/test.txt', help='List of test video ids')
    parser.add_argument('--test_data_loc', type=str, default='/scratch/shared/beegfs/hbull/shared-datasets/mediapi-rgb/videos/features', help='Location of test data')
    parser.add_argument('--test_labels_loc', type=str, default='/scratch/shared/beegfs/hbull/shared-datasets/mediapi-rgb/videos/subtitles.pkl', help='Location of test data')
    
    parser.add_argument('--stride', type=int, default=25, help='Pad inputs to this length')
    parser.add_argument('--input_len', type=int, default=25, help='Pad inputs to this length')

    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for rounding softmax outputs')
    parser.add_argument('--pos_weight', type=float, default=1, help='pos weight')
    parser.add_argument('--ff_size', type=int, default=2048, help='ff size transformer')

    parser.add_argument('--remove_labels_all_zero', type=bool, default=False, help='remove labels which are all zero')

    parser.add_argument('--print_text_output', type=str, default="NO SIGNING", help='text to print on the vtt file')
    parser.add_argument('--save_output_folder', type=str, default="no_signing_intervals_vtt", help='save output folder')

    parser.add_argument('--pad_input', type=int, default=16, help='Pad inputs to this length')
    parser.add_argument('--pad_output', type=int, default=5, help='Pad outputs to this length')

    parser.add_argument('--model', type=str, default='mlp', help='Model type')
    parser.add_argument('--dataset', type=str, default='features', help='Dataset type')
    parser.add_argument('--trainer', type=str, default='trainer', help='Trainer type')

    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='Grad clip norm')
 
    args = parser.parse_args()

    return args