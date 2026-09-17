from pathlib import Path


PCD_ROOT = Path(__file__).resolve().parent


OPENVLA_RESET_FREQUENCY = {
    "google_robot_move_near": 8,
    "google_robot_open_drawer" : 24,
    "google_robot_close_drawer" : 32,
    "google_robot_pick_coke_can" : 12, 
    "widowx_carrot_on_plate" : 8,
    "widowx_spoon_on_towel" : 8,
    "widowx_put_eggplant_in_basket" : 8,
    "widowx_stack_cube" : 4,
    "google_robot_place_apple_in_closed_top_drawer" : 8,
    "google_robot_pick_coke_can_drawer_variant": 12,
    "google_robot_pick_coke_can_light_variant": 12,
    "google_robot_pick_coke_can_dark": 12,
    "google_robot_pick_coke_can_table_paper_variant": 12,
    "google_robot_pick_coke_can_table_stone_variant": 12,
}

OPEN_PIZERO_RESET_FREQUENCY = {
    "google_robot_move_near": 12,
    "google_robot_open_drawer" : 12,
    "google_robot_close_drawer" : 8,
    "google_robot_pick_coke_can" : 16, 
    "widowx_carrot_on_plate" : 8,
    "widowx_spoon_on_towel" : 8,
    "widowx_put_eggplant_in_basket" : 16,
    "widowx_stack_cube" : 4,
    "google_robot_place_apple_in_closed_top_drawer" : 8,
}

# GR00T keeps a separate PCD-Fast layer-reselection schedule so it can be tuned
# independently from Pi0.
GR00T_RESET_FREQUENCY = {
    "google_robot_move_near": 8,
    "google_robot_open_drawer": 8,
    "google_robot_close_drawer": 8,
    "google_robot_pick_coke_can": 8,
    "widowx_carrot_on_plate": 8,
    "widowx_spoon_on_towel": 8,
    "widowx_put_eggplant_in_basket": 8,
    "widowx_stack_cube": 8,
    "google_robot_place_apple_in_closed_top_drawer": 8,
}

OPENVLA_CONFIG = dict(
    saved_model_path='openvla-7b',
    unnorm_key=None,
    policy_setup='google_robot',
    horizon=1,
    pred_action_horizon=1,
    exec_horizon=1,
    image_size=[224, 224],
    action_scale=1.0,
)

OPEN_PIZERO_CONFIG = dict(
    cfg_dir=str(PCD_ROOT / 'configs' / 'pi0'),
    use_ddp=False,
    use_naive=False,
    use_torch_compile=True,
)

GR00T_SERVER_HOST = "127.0.0.1"
GR00T_SERVER_BASE_PORT = 5555

GR00T_CONFIG = dict(
    server_address=f'tcp://{GR00T_SERVER_HOST}:{GR00T_SERVER_BASE_PORT}',
    timeout_ms=120000,
    policy_setup='google_robot',
    method='baseline',
    exec_horizon=1,
    alpha=0.0,
    reset_frequency=4,
    early_exit_layer=-1,
)

GR00T_PCD_MASK_CONFIG = dict(
    timeout_ms=1200000,
    method='pcd_mask',
    alpha=0.2,
    num_repeats=24,
    bandwidth_factor=1.0,
    keep_threshold=0.5,
)

GR00T_PCD_FAST_CONFIG = dict(
    timeout_ms=600000,
    method='pcd_fast',
    alpha=0.2,
    reset_frequency=4,
    early_exit_layer=-1,
)

CONTRAST_IMAGE_CONFIG = dict(
    camera_name=None,
    by="gt",
    inpaint_mode="lama",
    version=2,
    get_all_parts=False,
)

SEGMENTED_IMAGE_CONFIG = dict(
    camera_name=None,
    by="gt",
    version=2,
    get_all_parts=False,
)

OPENVLA_PCD_MASK_CONFIG = dict(
    alpha=0.8,
)

PIZERO_PCD_MASK_CONFIG = dict(
    alpha=0.2,
    num_repeats=24,
    bandwidth_factor=1.0,
    keep_threshold=0.5,
)

OPENVLA_CONFIG_EARLY_EXIT = dict(
    early_exit_layer=32,
)

OPENVLA_PCD_FAST_CONFIG = dict(
    early_exit_layer=32,
    alpha=0.8,
)

OPEN_PIZERO_EARLY_EXIT = dict(
    early_exit_layer=18,
)

PIZERO_PCD_FAST_CONFIG = dict(
    early_exit_layer=18,
    alpha=0.2,
)


def get_policy_config(policy, checkpoint, task, opts, contrast, early_exit):
    if policy == 'openvla':
        config = dict(OPENVLA_CONFIG)
        config['saved_model_path'] = checkpoint
    elif policy == 'pizero':
        config = dict(OPEN_PIZERO_CONFIG)
        config['checkpoint_path'] = checkpoint
    elif policy == 'gr00t':
        config = dict(GR00T_CONFIG)
        config['model_path'] = checkpoint
    else:
        raise NotImplementedError()
    
    # select policy setup based on task
    if task.startswith('google_robot'):
        config['policy_setup'] = 'google_robot'
        if policy == 'gr00t':
            config['exec_horizon'] = 1
    elif task.startswith('widowx'):
        config['policy_setup'] = 'widowx_bridge'
        if policy == 'gr00t':
            config['exec_horizon'] = 4
    else:
        raise NotImplementedError
    
    if early_exit and not contrast:
        if policy == 'openvla':
            config.update(OPENVLA_CONFIG_EARLY_EXIT)
        elif policy == 'pizero':
            config.update(OPEN_PIZERO_EARLY_EXIT)
        else:
            raise NotImplementedError()

    # update config if contrast policy is used
    if contrast and not early_exit:
        if policy == 'openvla':
            config.update(OPENVLA_PCD_MASK_CONFIG)
        elif policy == 'pizero':
            config.update(PIZERO_PCD_MASK_CONFIG)
        elif policy == 'gr00t':
            config.update(GR00T_PCD_MASK_CONFIG)
        else:
            raise NotImplementedError()
        
    if early_exit and contrast:
        if policy == 'openvla':
            config.update(OPENVLA_PCD_FAST_CONFIG)
            config['reset_frequency'] = OPENVLA_RESET_FREQUENCY.get(task, 4)
        elif policy == 'pizero':
            config.update(PIZERO_PCD_FAST_CONFIG)
            config['reset_frequency'] = OPEN_PIZERO_RESET_FREQUENCY.get(task, 4)
        elif policy == 'gr00t':
            config.update(GR00T_PCD_FAST_CONFIG)
            config['reset_frequency'] = GR00T_RESET_FREQUENCY.get(
                task, GR00T_PCD_FAST_CONFIG['reset_frequency']
            )
        else:
            raise NotImplementedError()
    
    # update opts
    for k, v in opts.items():
        if k in config:
            config[k] = v
    
    return config


def get_contrast_image_generator_config(opts):
    config = dict(CONTRAST_IMAGE_CONFIG)
    for k, v in opts.items():
        if k in config:
            config[k] = v
    return config

def get_segmented_image_generator_config(opts):
    config = dict(SEGMENTED_IMAGE_CONFIG)
    for k, v in opts.items():
        if k in config:
            config[k] = v
    return config
