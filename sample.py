import random
import argparse
import os
import torch as th
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import cv2
from matplotlib import pyplot as plt

from sketch_diffusion import logger
from sketch_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    # different modes
    create_model_and_diffusion,
    # create_model_and_diffusion_acc
    # create_model_and_diffusion_noise,
    add_dict_to_argparser,
    args_to_dict,
)


def canvas_size_google(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)  

    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)

    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  
    start_y = -ymin - sketch[0][1]
    return [int(start_x), int(start_y), int(h), int(w)]


def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=float)
    return sketch_rescale.astype("int16")


def draw_three(sketch, window_name="google", padding=30, random_color=False, time=1, show=False, img_size=256):
    """
    实际上是隔一个点断开
    """
    thickness = int(img_size * 0.025)

    sketch = scale_sketch(sketch, (img_size, img_size))  
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1
    canvas = np.ones((max(h, w) + 3 * (thickness + 1), max(h, w) + 3 * (thickness + 1), 3), dtype='uint8') * 255
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (0, 0, 0)
    pen_now = np.array([start_x, start_y])
    first_zero = False
    for stroke in sketch:
        delta_x_y = stroke[: 2]
        state = stroke[2:]
        if first_zero:  
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state.item()) == 1:
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (0, 0, 0)
        pen_now += delta_x_y
    return cv2.resize(canvas, (img_size, img_size))


def bin_pen(x, pen_break=0.005):
    result = x
    for i in range(x.size()[0]):
        for j in range(x.size()[1]):
                pen = x[i][j][2]
                if pen >= pen_break:
                    result[i][j][2] = 1
                else:
                    result[i][j][2] = 0
    return result[:, :, :3]


def main():
    args = create_argparser().parse_args()

    os.makedirs(args.log_dir + '/test', exist_ok=True)
    args.log_dir = args.log_dir + '/test'
    os.makedirs(args.save_path, exist_ok=True)

    logger.configure(args.log_dir)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.cuda()
    model.eval()

    logger.log("sampling...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device='cuda'
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, pen_state, _ = sample_fn(
            model,
            (args.batch_size, args.image_size, 2),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample_all = th.cat((sample, pen_state), 2).cpu()
        sample_all = bin_pen(sample_all, args.pen_break)

        save_idx = 0
        for sample in sample_all:
            sample = sample.numpy()

            sketch_cv = draw_three(sample, img_size=256)
            tensor = transforms.ToTensor()(sketch_cv)
            all_images.append(tensor)

            save_sketch_sketchknitter(sample, os.path.join(args.save_path, f'output_{save_idx}.png'))
            save_idx += 1


            # rel_coors = sample[:, :2]
            # abs_coor = np.cumsum(rel_coors, axis=0)
            # sample[:, :2] = abs_coor
            #
            # strokes = np.split(sample, np.where(sample[:, 2] == 0)[0] + 1)
            # for c_stk in strokes:
            #     plt.plot(c_stk[:, 0], -c_stk[:, 1])
            #
            # plt.savefig(os.path.join(args.save_path, f'output_{save_idx}.png'))
            # plt.clf()
            # plt.close()




        # np.savez(os.path.join(args.save_path, 'result.npz'), sample_all)

    save_image(th.stack(all_images), os.path.join(args.save_path, 'output.png'))


def save_sketch(sample, save_path):
    rel_coors = sample[:, :2]
    abs_coor = np.cumsum(rel_coors, axis=0)
    sample[:, :2] = abs_coor

    strokes = np.split(sample, np.where(sample[:, 2] == 0)[0] + 1)
    for c_stk in strokes:
        plt.plot(c_stk[:, 0], -c_stk[:, 1])

    plt.axis('equal')
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def save_sketch_sketchknitter(sample, save_path):
    """
    实际上是隔一个点空一段
    """
    rel_coors = sample[:, :2]
    abs_coor = np.cumsum(rel_coors, axis=0)

    # 将草图固定为偶数个点
    if len(abs_coor) % 2 != 0:
        abs_coor = abs_coor[:-1, :]

    abs_coor = abs_coor[1: -1, :]

    for i in range(len(abs_coor) // 2):
        plt.plot(abs_coor[2 * i: 2 * i + 2, 0], -abs_coor[2 * i: 2 * i + 2, 1])

    plt.axis('equal')
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50,
        batch_size=16,
        use_ddim=True,
        model_path=r'model_trained/sketchknitter.pth',
        log_dir=r'./logs',
        save_path=r'./imgs_gen',
        pen_break=0.1,
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

