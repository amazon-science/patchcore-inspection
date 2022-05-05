import pytest
import torch
import torch.utils.data

from patchcore import sampler


def _dummy_features(feature_dimension):
    num_samples = 5000
    return (
        torch.arange(num_samples).unsqueeze(1)
        / float(num_samples)
        * torch.ones((num_samples, feature_dimension))
    )


def _dummy_constant_features(number_of_examples, feature_dimension):
    return torch.ones([number_of_examples, feature_dimension])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fails for non-GPU machine.")
def test_standard_greedy_coreset_sampling():
    feature_dimension = 2
    init_features = _dummy_features(feature_dimension)

    sampling_percentage = 0.1
    model = sampler.GreedyCoresetSampler(
        percentage=sampling_percentage,
        device=torch.device("cpu"),
        dimension_to_project_features_to=feature_dimension,
    )
    subsampled_features = model.run(init_features)

    target_num_subsampled_features = int(len(init_features) * sampling_percentage)
    assert len(subsampled_features) == target_num_subsampled_features
    assert (
        len(torch.unique(subsampled_features, dim=0)) == target_num_subsampled_features
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fails for non-GPU machine.")
def test_approximate_greedy_coreset_sampling():
    feature_dimension = 2
    init_features = _dummy_features(feature_dimension)

    sampling_percentage = 0.1
    model = sampler.ApproximateGreedyCoresetSampler(
        percentage=sampling_percentage,
        device=torch.device("cpu"),
        number_of_starting_points=10,
        dimension_to_project_features_to=feature_dimension,
    )
    subsampled_features = model.run(init_features)

    target_num_subsampled_features = int(len(init_features) * sampling_percentage)
    assert len(subsampled_features) == target_num_subsampled_features
    assert (
        len(torch.unique(subsampled_features, dim=0)) == target_num_subsampled_features
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fails for non-GPU machine.")
def test_coreset_sampling_on_same_samples():
    feature_dimension = 2
    init_features = _dummy_constant_features(5000, feature_dimension)

    sampling_percentage = 0.1
    model = sampler.ApproximateGreedyCoresetSampler(
        percentage=sampling_percentage,
        device=torch.device("cpu"),
        number_of_starting_points=10,
        dimension_to_project_features_to=feature_dimension,
    )
    subsampled_features = model.run(init_features)

    target_num_subsampled_features = int(len(init_features) * sampling_percentage)
    assert len(subsampled_features) == target_num_subsampled_features
    assert len(torch.unique(subsampled_features, dim=0)) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fails for non-GPU machine.")
def test_random_sampling():
    init_features = _dummy_features(feature_dimension=2)

    sampling_percentage = 0.1
    model = sampler.RandomSampler(percentage=sampling_percentage)
    subsampled_features = model.run(init_features)

    target_num_subsampled_features = int(len(init_features) * sampling_percentage)
    assert len(subsampled_features) == target_num_subsampled_features
    assert (
        len(torch.unique(subsampled_features, dim=0)) == target_num_subsampled_features
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fails for non-GPU machine.")
def test_type_retention():
    feature_dimension = 2
    init_features = _dummy_features(feature_dimension)

    sampling_percentage = 0.1
    model = sampler.ApproximateGreedyCoresetSampler(
        percentage=sampling_percentage,
        device=torch.device("cpu"),
        number_of_starting_points=10,
        dimension_to_project_features_to=feature_dimension,
    )
    subsampled_features = model.run(init_features)

    assert type(subsampled_features) == type(init_features)
    assert subsampled_features.device == init_features.device

    subsampled_features_numpy = model.run(init_features.numpy())
    assert isinstance(subsampled_features_numpy, type(init_features.numpy()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fails for non-GPU machine.")
def test_johnsonlindenstrauss_reduction():
    feature_dimension = 256
    init_features = _dummy_features(feature_dimension)

    sampling_percentage = 0.1
    dimension_to_project_features_to = 64

    model = sampler.ApproximateGreedyCoresetSampler(
        percentage=sampling_percentage,
        device=torch.device("cpu"),
        number_of_starting_points=10,
        dimension_to_project_features_to=dimension_to_project_features_to,
    )
    subsampled_features = model.run(init_features)

    assert subsampled_features.shape[-1] == feature_dimension
