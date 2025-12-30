import torch
import os


def save_checkpoint(model, optimizer, epoch, path, is_best=False, extra_state=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'epoch': epoch
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


def load_checkpoint(model, path, device='cpu', optimizer=None, strict=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        epoch = checkpoint.get('epoch', 0)
        if optimizer and 'optimizer' in checkpoint and checkpoint['optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        state_dict = checkpoint
        epoch = 0

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)

    # Extract extra state if available
    extra_state = {}
    for key in checkpoint:
        if key not in ['model', 'optimizer', 'epoch']:
            extra_state[key] = checkpoint[key]

    return model, optimizer, epoch, extra_state


def load_teacher_model(model, checkpoint_path, device='cpu', strict=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if strict:
            raise
        print(f"Warning: Loading weights with strict=False due to architecture mismatch")
        model.load_state_dict(state_dict, strict=False)

    return model
