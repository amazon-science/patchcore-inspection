import numpy as np
import torch.utils.data
from torchvision import models

import patchcore
from patchcore import patchcore as patchcore_model


def _dummy_features(number_of_examples, shape_of_examples):
    return torch.Tensor(
        np.stack(number_of_examples * [np.ones(shape_of_examples)], axis=0)
    )


def _dummy_constant_dataloader(number_of_examples, shape_of_examples):
    features = _dummy_features(number_of_examples, shape_of_examples)
    return torch.utils.data.DataLoader(features, batch_size=1)


def _dummy_various_features(number_of_examples, shape_of_examples):
    images = torch.ones((number_of_examples, *shape_of_examples))
    multiplier = torch.arange(number_of_examples) / float(number_of_examples)
    for _ in range(images.ndim - 1):
        multiplier = multiplier.unsqueeze(-1)
    return multiplier * images


def _dummy_various_dataloader(number_of_examples, shape_of_examples):
    features = _dummy_various_features(number_of_examples, shape_of_examples)
    return torch.utils.data.DataLoader(features, batch_size=1)


def _dummy_images(number_of_examples, image_shape):
    torch.manual_seed(0)
    return torch.rand([number_of_examples, *image_shape])


def _dummy_image_random_dataloader(number_of_examples, image_shape):
    images = _dummy_images(number_of_examples, image_shape)
    return torch.utils.data.DataLoader(images, batch_size=4)


def _standard_patchcore(image_dimension):
    patchcore_instance = patchcore_model.PatchCore(torch.device("cpu"))
    backbone = models.wide_resnet50_2(pretrained=False)
    backbone.name, backbone.seed = "wideresnet50", 0
    patchcore_instance.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        input_shape=[3, image_dimension, image_dimension],
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        spade_nn=2,
    )
    return patchcore_instance


def _load_patchcore_from_path(load_path):
    patchcore_instance = patchcore_model.PatchCore(torch.device("cpu"))
    patchcore_instance.load_from_path(
        load_path=load_path,
        device=torch.device("cpu"),
        prepend="temp_patchcore",
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    return patchcore_instance


def _approximate_greedycoreset_sampler_with_reduction(
    sampling_percentage, johnsonlindenstrauss_dim
):
    return patchcore.sampler.ApproximateGreedyCoresetSampler(
        percentage=sampling_percentage,
        device=torch.device("cpu"),
        number_of_starting_points=10,
        dimension_to_project_features_to=johnsonlindenstrauss_dim,
    )


def test_dummy_patchcore():
    image_dimension = 112
    model = _standard_patchcore(image_dimension)
    training_dataloader = _dummy_constant_dataloader(
        4, [3, image_dimension, image_dimension]
    )
    print(model.featuresampler)
    model.fit(training_dataloader)

    test_features = torch.Tensor(2 * np.ones([2, 3, image_dimension, image_dimension]))
    scores, masks = model.predict(test_features)

    assert all([score > 0 for score in scores])
    for mask in masks:
        assert np.all(mask.shape == (image_dimension, image_dimension))


def test_patchcore_on_dataloader():
    """Test PatchCore on dataloader and assure training scores are zero."""
    image_dimension = 112
    model = _standard_patchcore(image_dimension)

    training_dataloader = _dummy_constant_dataloader(
        4, [3, image_dimension, image_dimension]
    )
    model.fit(training_dataloader)
    scores, masks, labels_gt, masks_gt = model.predict(training_dataloader)

    assert all([score < 1e-3 for score in scores])
    for mask, mask_gt in zip(masks, masks_gt):
        assert np.all(mask.shape == (image_dimension, image_dimension))
        assert np.all(mask_gt.shape == (image_dimension, image_dimension))


def test_patchcore_load_and_saveing(tmpdir):
    image_dimension = 112
    model = _standard_patchcore(image_dimension)

    training_dataloader = _dummy_constant_dataloader(
        4, [3, image_dimension, image_dimension]
    )
    model.fit(training_dataloader)
    model.save_to_path(tmpdir, "temp_patchcore")

    test_features = torch.Tensor(
        1.234 * np.ones([2, 3, image_dimension, image_dimension])
    )
    scores, masks = model.predict(test_features)
    other_scores, other_masks = model.predict(test_features)

    assert np.all(scores == other_scores)
    for mask, other_mask in zip(masks, other_masks):
        assert np.all(mask == other_mask)


def test_patchcore_real_data():
    image_dimension = 112
    sampling_percentage = 0.1
    model = _standard_patchcore(image_dimension)
    model.sampler = _approximate_greedycoreset_sampler_with_reduction(
        sampling_percentage=sampling_percentage,
        johnsonlindenstrauss_dim=64,
    )

    num_dummy_train_images = 50
    training_dataloader = _dummy_various_dataloader(
        num_dummy_train_images, [3, image_dimension, image_dimension]
    )
    model.fit(training_dataloader)

    num_dummy_test_images = 5
    test_dataloader = _dummy_various_dataloader(
        num_dummy_test_images, [3, image_dimension, image_dimension]
    )
    scores, masks, labels_gt, masks_gt = model.predict(test_dataloader)

    for mask, mask_gt in zip(masks, masks_gt):
        assert np.all(mask.shape == (image_dimension, image_dimension))
        assert np.all(mask_gt.shape == (image_dimension, image_dimension))

    assert len(scores) == 5
