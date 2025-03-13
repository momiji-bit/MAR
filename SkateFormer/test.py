from main import *

parser = get_parser()

# load arg form config file
p = parser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.safe_load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
parser.set_defaults(**default_arg)

arg = parser.parse_args()
init_seed(arg.seed)
processor = Processor(arg)