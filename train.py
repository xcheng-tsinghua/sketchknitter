import argparse
import os
import torch

from sketch_diffusion import logger
from sketch_diffusion.image_datasets import load_data
from sketch_diffusion.resample import create_named_schedule_sampler
from sketch_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion, # you can change mode here
    args_to_dict,
    str2bool,
)
from sketch_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    print(args)

    model_dir = './model_trained'
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger.configure(args.log_dir)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(  # you can change mode here
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model_save_path = os.path.join(model_dir, f'sketchknitter_{args.category}.pth')
    try:
        model.load_state_dict(torch.load(model_save_path))
        print('training from exist model: ' + model_save_path)
    except:
        print('no existing model, training from scratch')

    model.cuda()
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    if args.local == 'True':
        data_dir = args.data_dir_local
    elif args.local == 'False':
        data_dir = args.data_dir_sever
    else:
        raise TypeError('error local type')

    data = load_data(
        data_dir=data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        # category=["sketchrnn_apple.full.npz"],  # replace your own datasets name.
        category=[args.category + '.full.npz'],  # replace your own datasets name.
        class_cond=False,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_save_path=model_save_path
    ).run_loop()

    # Reuse the final trained model for all five sampling runs.
    from sample import generate_samples_archive

    model.float()
    generate_samples_archive(
        model, diffusion, category=args.category, image_size=args.image_size,
        class_cond=args.class_cond,
    )


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_sever", type=str, default='/root/my_data/data_set/quickdraw/raw')
    parser.add_argument("--data_dir_local", type=str, default='D:\\document\\DeepLearning\\DataSet\\quickdraw\\raw')
    parser.add_argument("--local", type=str, default='False')
    parser.add_argument("--category", type=str, default='apple')
    parser.add_argument("--schedule_sampler", type=str, default='uniform')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_anneal_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--microbatch", type=int, default=-1)
    parser.add_argument("--ema_rate", type=str, default='0.9999')
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--resume_checkpoint", type=str, default='')
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--fp16_scale_growth", type=float, default=0.001)
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--num_channels", type=int, default=96)
    parser.add_argument("--num_res_blocks", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_heads_upsample", type=int, default=-1)
    parser.add_argument("--attention_resolutions", type=str, default='16,8')
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learn_sigma", type=str2bool, default=False)
    parser.add_argument("--sigma_small", type=str2bool, default=False)
    parser.add_argument("--class_cond", type=str2bool, default=False)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default='linear')
    parser.add_argument("--timestep_respacing", type=str, default='')
    parser.add_argument("--use_kl", type=str2bool, default=False)
    parser.add_argument("--predict_xstart", type=str2bool, default=False)
    parser.add_argument("--rescale_timesteps", type=str2bool, default=True)
    parser.add_argument("--rescale_learned_sigmas", type=str2bool, default=True)
    parser.add_argument("--use_checkpoint", type=str2bool, default=False)
    parser.add_argument("--use_scale_shift_norm", type=str2bool, default=True)
    return parser


if __name__ == "__main__":
    main()
