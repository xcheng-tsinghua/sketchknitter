import random
import argparse
import os
import io
import zipfile
import torch as th
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
    str2bool,
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


SAMPLING_SEEDS = (92, 48, 69, 83, 5)


def generate_samples_archive(
    model, diffusion, *, category, image_size, class_cond=False,
    save_path='./imgs_gen', num_samples=1000, batch_size=20,
    use_ddim=True, clip_denoised=True, pen_break=0.1,
):
    """Render all seeds into an in-memory ZIP, then save the archive once."""
    if batch_size <= 0 or num_samples <= 0:
        raise ValueError('batch_size and num_samples must be positive')
    model.eval()
    device = next(model.parameters()).device
    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    archive_path = os.path.join(save_path, f'{category}_seeds_92_48_69_83_5.zip')
    with io.BytesIO() as archive_buffer:
        with zipfile.ZipFile(archive_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            with th.no_grad():
                for seed in SAMPLING_SEEDS:
                    random.seed(seed)
                    np.random.seed(seed)
                    th.manual_seed(seed)
                    th.cuda.manual_seed_all(seed)
                    th.backends.cudnn.benchmark = False
                    th.backends.cudnn.deterministic = True
                    for start in range(0, num_samples, batch_size):
                        count = min(batch_size, num_samples - start)
                        print(f'seed {seed}: sampling [{start} to {start + count})')
                        model_kwargs = {}
                        if class_cond:
                            model_kwargs['y'] = th.randint(
                                low=0, high=NUM_CLASSES, size=(count,), device=device,
                            )
                        sample, pen_state, _ = sample_fn(
                            model, (count, image_size, 2),
                            clip_denoised=clip_denoised, model_kwargs=model_kwargs,
                        )
                        sketches = bin_pen(th.cat((sample, pen_state), 2).cpu(), pen_break)
                        for index, sketch in enumerate(sketches, start=start):
                            with io.BytesIO() as image_buffer:
                                save_sketch_sketchknitter(sketch.numpy(), image_buffer)
                                archive.writestr(
                                    f'{category}/seed_{seed}/output_{index}.png',
                                    image_buffer.getvalue(),
                                )
        os.makedirs(save_path, exist_ok=True)
        with open(archive_path, 'wb') as output:
            with archive_buffer.getbuffer() as archive_data:
                output.write(archive_data)
    print(f'Saved {len(SAMPLING_SEEDS) * num_samples} sketches to {archive_path}')
    return archive_path


def main():
    args = create_argparser().parse_args()
    logger.configure(os.path.join(args.log_dir, 'test'))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    weight_path = f'./model_trained/sketchknitter_{args.category}.pth'
    model.load_state_dict(th.load(weight_path))
    print(f'loading weights from {weight_path}')
    model.cuda()
    generate_samples_archive(
        model, diffusion, category=args.category, image_size=args.image_size,
        class_cond=args.class_cond, save_path=args.save_path,
        num_samples=args.num_samples, batch_size=args.batch_size,
        use_ddim=args.use_ddim, clip_denoised=args.clip_denoised,
        pen_break=args.pen_break,
    )


def save_sketch(sample, save_path):
    rel_coors = sample[:, :2]
    abs_coor = np.cumsum(rel_coors, axis=0)
    sample[:, :2] = abs_coor

    strokes = np.split(sample, np.where(sample[:, 2] == 0)[0] + 1)
    for c_stk in strokes:
        plt.plot(c_stk[:, 0], -c_stk[:, 1])

    plt.axis('equal')
    plt.savefig(save_path, format='png')
    plt.clf()
    plt.close()


def save_sketch_sketchknitter(sample, save_path, linewidth=5):
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
        plt.plot(abs_coor[2 * i: 2 * i + 2, 0], -abs_coor[2 * i: 2 * i + 2, 1], linewidth=linewidth)

    plt.axis('equal')
    plt.axis('off')
    plt.savefig(save_path, format='png')
    plt.clf()
    plt.close()


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_denoised", type=str2bool, default=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--use_ddim", type=str2bool, default=True)
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--save_path", type=str, default='./imgs_gen')
    parser.add_argument("--pen_break", type=float, default=0.1)
    parser.add_argument("--category", type=str, default='moon')
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

