import torch


@torch.no_grad()
def class_activation_map(features, fc_w, class_idx):
    """
    Refer to [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)
    :param features: the features from a model, where the shape is `torch.Size([B, d, h, w])`
    :param fc_w: the weight of fully-connected layers from a model, where the shape is `torch.Size([num_classes, d])`
    :param class_idx: the index of specific class, where the shape is `torch.Size([B, K])`
    :return: CAM: the shape is `torch.Size([B, K, h, w])`
    """
    if features.dim() != 4:
        raise RuntimeError(f'The dimension of `features` should be 4D, got {features.dim()}D.')

    if fc_w.dim() != 2:
        raise RuntimeError(f'The dimension of `fc_w` should be 2D, got {fc_w.dim()}D.')

    if class_idx.dim() != 2:
        raise RuntimeError(f'The dimension of `class_idx` should be 2D, got {class_idx.dim()}D.')

    v_classes = []
    for batch in range(class_idx.shape[0]):
        v_class = fc_w[class_idx[batch]]  # the shape of `v_class` is (K, d)
        v_classes.append(v_class)

    v_classes = torch.stack(v_classes)  # the shape of `v_classes` is (B, K, d)
    v_classes = v_classes.reshape(*v_classes.shape[:3], 1, 1)  # (B, K, d, 1, 1)
    features = features.unsqueeze(dim=1)  # (B, 1, d, h, w)
    return torch.sum(v_classes * features, dim=2)
