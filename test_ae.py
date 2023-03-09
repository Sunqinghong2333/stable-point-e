import os
import time
import argparse
import torch
from tqdm.auto import tqdm

# from utils.dataset import *
from utils.dataset_new import *
from utils.misc import *
from utils.data import *
# from models.autoencoder import *
from models.autoencoder_new import *
from models.clipencoder import *
from evaluation import EMD_CD


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/AE_airplane.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--clip_name', type=str, default="ViT-L/14")
parser.add_argument('--clip_root', type=str, default="/mnt/cache/sunqinghong1/workspace/point-e/point_e/examples/point_e_model_cache")
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=96)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
# model.load_state_dict(ckpt['state_dict'])
model_clip = CLIPEncoder(args)

all_ref = []
all_recons = []
all_cat = []
for i, batch in enumerate(tqdm(test_loader)):
    ref = batch['pointcloud'].to(args.device)
    y = batch['cate']
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    model_clip.eval()
    with torch.no_grad():
        code = model_clip.encode(y)
        recons = model.decode(code, y, ref.size(1)).detach()

    # import ipdb
    # ipdb.set_trace()

    ref = ref * scale + shift
    recons = recons.permute(0, 2, 1) * scale + shift

    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())
    all_cat = all_cat + y

    break

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())
np.save(os.path.join(save_dir, 'cat.npy'), np.array(all_cat))

# logger.info('Start computing metrics...')
# metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
# cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
# logger.info('CD:  %.12f' % cd)
# logger.info('EMD: %.12f' % emd)
