import argparse
import os

from sketch_diffusion import logger
from sketch_diffusion.image_datasets import load_data
from sketch_diffusion.resample import create_named_schedule_sampler
from sketch_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion, # you can change mode here
    args_to_dict,
    add_dict_to_argparser,
)
from sketch_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    print(args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # dist_util.setup_dist()
    logger.configure(args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion( # you can change mode here
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

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
        category=[args.category],  # replace your own datasets name.
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
        model_save_path=args.model_save_path
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir_sever=fr'/root/my_data/data_set/quickdraw/raw',
        data_dir_local=r'D:\document\DeepLearning\DataSet\quickdraw\raw',
        local='False',
        category='apple.full.npz',  # e.g. apple.full.npz
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1024,
        microbatch=-1,  
        ema_rate="0.9999",  
        log_interval=5,
        save_interval=10,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir='./logs',
        model_save_path='model_trained/sketchknitter.pth'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    os.makedirs('./imgs_gen', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./model_trained', exist_ok=True)
    main()
