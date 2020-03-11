import os
import random
import time

import numba
import numpy as np


def evaluate(configs):
    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ###########
    # Prepare #
    ###########

    if configs.device == 'cuda':
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False
    if ('seed' not in configs) or (configs.seed is None):
        configs.seed = torch.initial_seed() % (2 ** 32 - 1)

    #################################
    # Initialize DataLoaders, Model #
    #################################
    print(f'\n==> loading dataset "{configs.dataset}"')
    dataset = configs.dataset()[configs.dataset.split]

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)

    if os.path.exists(configs.evaluate.best_checkpoint_path):
        print(f'==> loading checkpoint "{configs.evaluate.best_checkpoint_path}"')
        checkpoint = torch.load(configs.evaluate.best_checkpoint_path)
        model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    model.eval()
    seed = random.randint(1, int(time.time())) % (2 ** 32 - 1)

    loader = DataLoader(
        dataset, shuffle=False, batch_size=configs.evaluate.batch_size,
        num_workers=configs.data.num_workers, pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )

    ##############
    # Evaluation #
    ##############

    predictions = np.zeros((len(dataset), 7))
    size_templates = configs.data.size_templates.to(configs.device)
    heading_angle_bin_centers = torch.arange(
        0, 2 * np.pi, 2 * np.pi / configs.data.num_heading_angle_bins).to(configs.device)
    current_step = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='eval', ncols=0):
            for k, v in inputs.items():
                inputs[k] = v.to(configs.device, non_blocking=True)
            outputs = model(inputs)

            center = outputs['center']  # (B, 3)
            heading_scores = outputs['heading_scores']  # (B, NH)
            heading_residuals = outputs['heading_residuals']  # (B, NH)
            size_scores = outputs['size_scores']  # (B, NS)
            size_residuals = outputs['size_residuals']  # (B, NS, 3)

            batch_size = center.size(0)
            batch_id = torch.arange(batch_size, device=center.device)
            heading_bin_id = torch.argmax(heading_scores, dim=1)
            heading = heading_angle_bin_centers[heading_bin_id] + heading_residuals[batch_id, heading_bin_id]  # (B, )
            size_template_id = torch.argmax(size_scores, dim=1)
            size = size_templates[size_template_id] + size_residuals[batch_id, size_template_id]  # (B, 3)

            center = center.cpu().numpy()
            heading = heading.cpu().numpy()
            size = size.cpu().numpy()
            rotation_angle = targets['rotation_angle'].cpu().numpy()  # (B, )

            update_predictions(predictions=predictions, center=center, heading=heading,
                               size=size, rotation_angle=rotation_angle,
                               current_step=current_step, batch_size=batch_size)
            current_step += batch_size

    np.save(configs.evaluate.stats_path, predictions)


@numba.jit()
def update_predictions(predictions, center, heading, size, rotation_angle, current_step, batch_size):
    for b in range(batch_size):
        l, w, h = size[b]
        x, y, z = center[b]  # (3)
        r = rotation_angle[b]
        t = heading[b]
        v_cos = np.cos(r)
        v_sin = np.sin(r)
        cx = v_cos * x + v_sin * z  # it should be v_cos * x - v_sin * z, but the rotation angle = -r
        cy = y + h / 2.0
        cz = v_cos * z - v_sin * x  # it should be v_sin * x + v_cos * z, but the rotation angle = -r
        r = r + t
        while r > np.pi:
            r = r - 2 * np.pi
        while r < -np.pi:
            r = r + 2 * np.pi
        predictions[current_step + b] = [h, w, l, cx, cy, cz, r]


