import torch
import copy
from InstantID import (
    Resampler,
    AttnProcessor,
    IPAttnProcessor
)
from diffusers import ControlNetModel
import gc
# ControlNet 모델을 unet으로부터 생성하는 함수
def set_controlnet(unet, 
                   face_ckpt_path=None, 
                   tile_ckpt_path=None,
                   depth_ckpt_path=None,
                   device="cuda", 
                   dtype=torch.float16):
    controlnet_base = ControlNetModel.from_unet(unet)
    if face_ckpt_path is not None:
        controlnet_face = copy.deepcopy(controlnet_base)
        controlnet_face_state_dict = torch.load(face_ckpt_path, map_location="cpu")
        m, u = controlnet_face.load_state_dict(controlnet_face_state_dict['state_dict'])
        print(f'[CONTROLNET FACE] MISSING: {len(m)} | UNEXPECTED: {len(u)} |')
    if tile_ckpt_path is not None:
        controlnet_tile = copy.deepcopy(controlnet_base)
        controlnet_tile_state_dict = torch.load(tile_ckpt_path, map_location="cpu")
        m, u = controlnet_tile.load_state_dict(controlnet_tile_state_dict)
        print(f'[CONTROLNET TILE] MISSING: {len(m)} | UNEXPECTED: {len(u)} |')
    if depth_ckpt_path is not None:
        controlnet_depth = copy.deepcopy(controlnet_base)
        controlnet_depth_state_dict = torch.load(depth_ckpt_path, map_location="cpu")
        m, u = controlnet_depth.load_state_dict(controlnet_depth_state_dict)
        print(f'[CONTROLNET DEPTH] MISSING: {len(m)} | UNEXPECTED: {len(u)} |')

    del controlnet_base
    gc.collect()
    torch.cuda.empty_cache()
    return controlnet_face.to(device=device, dtype=dtype), controlnet_tile.to(device=device, dtype=dtype), controlnet_depth.to(device=device, dtype=dtype)

# 추출된 face feature를 unet 내부 연산에 사용되기 적합한 형태로 projection하는 모델
def set_feature_proj_model(ckpt_path, image_emb_dim=512, num_tokens=16, device="cuda", dtype=torch.float16):
    feature_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=image_emb_dim,
        output_dim=768,
        ff_mult=4,
    ).eval()
    feature_proj_state_dict = torch.load(ckpt_path, map_location="cpu")
    m, u = feature_proj_model.load_state_dict(feature_proj_state_dict['state_dict'])
    print(f'[FEATURE PROJ MODEL] MISSING: {len(m)} | UNEXPECTED: {len(u)} |')
    return feature_proj_model.to(device, dtype=dtype)

# IP-Adapater를 unet에 적용하는 함수
def set_ip_adapter_processors(unet, ckpt_path, num_tokens=16, scale=1.0, ignore_motion=False, device="cuda", dtype=torch.float16):
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if (cross_attention_dim is None) or (ignore_motion and 'motion_modules.' in name):
            attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
        else:
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                cross_attention_dim=cross_attention_dim, 
                                                scale=scale,
                                                num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)            
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    ip_adapter_state_dict = torch.load(ckpt_path, map_location="cpu")
    sd={}
    for name, param in ip_adapter_state_dict['state_dict'].items():
        sd[name.replace('module.', '')] = param
    m, u = unet.load_state_dict(sd, strict=False)
    print(f'[IP-ADAPTER] MISSING: {len(m)} | UNEXPECTED: {len(u)} |')
    return unet.to(device, dtype=dtype)

def set_ip_adapter_scale(unet, scale=1.0):
    for attn_processor in unet.attn_processors.values():
        if isinstance(attn_processor, IPAttnProcessor):
            attn_processor.scale = scale