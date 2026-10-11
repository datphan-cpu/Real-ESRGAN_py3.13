import torch
import os

from facexlib.utils import load_file_from_url
from .bisenet import BiSeNet
from .parsenet import ParseNet


def init_parsing_model(model_name='bisenet', half=False, device='cuda', model_rootpath=None):
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # === ĐOẠN SỬA LẠI TỪ DÒNG 18 ĐẾN 21 ===
    # Tự động xác định tên file trọng số dựa vào model_name
    # 1. Xác định tên file mặc định dựa theo model_name
    default_filename = 'parsing_bisenet.pth' if model_name == 'bisenet' else 'parsing_parsenet.pth'

    # 2. Xử lý logic đường dẫn thư mục lưu trữ weights
    if model_rootpath is None:
        dst_dir = os.path.join('facexlib', 'weights')
    elif os.path.isdir(model_rootpath):
        dst_dir = model_rootpath
    else:
        dst_dir = os.path.dirname(model_rootpath)

    os.makedirs(dst_dir, exist_ok=True)
    model_path = os.path.join(dst_dir, default_filename)

    # 3. TỰ ĐỘNG TẢI FILE: Kiểm tra và tải file weights từ URL nếu chưa có sẵn trên máy
    if not os.path.exists(model_path):
        print(f"⏳ Not found {default_filename}. Downloading from facexlib...")
        model_path = load_file_from_url(
            url=model_url,
            model_dir=dst_dir,
            progress=True
        )
    # ==========================================




    try:
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    except TypeError:
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
