import numpy as np
import pytest

from patchcore import common


def test_calling_without_setting_index():
    query = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
    index = 2 * query

    nn_search = common.FaissNN()

    distances_before_set_index, nn_indices_before_set_index = nn_search.run(
        2, query, index
    )
    nn_search.fit(index)
    distances_after_set_index, nn_indices_after_set_index = nn_search.run(2, query)

    assert np.all(distances_before_set_index == distances_after_set_index)
    assert np.all(nn_indices_before_set_index == nn_indices_after_set_index)


def test_approximate_faiss():
    query = np.ones([768, 128], dtype=np.float32)
    index = 2 * query

    nn_search = common.ApproximateFaissNN()

    distances_before_set_index, nn_indices_before_set_index = nn_search.run(
        2, query, index
    )
    nn_search.fit(index)
    distances_after_set_index, nn_indices_after_set_index = nn_search.run(2, query)

    assert np.all(distances_before_set_index == distances_after_set_index)
    assert np.all(nn_indices_before_set_index == nn_indices_after_set_index)


def test_search_without_index_raises_exception():
    features = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
    nn_search = common.FaissNN(on_gpu=False, num_workers=4)
    with pytest.raises(AttributeError):
        nn_search.run(2, features)
    assert nn_search.run(2, features, features) is not None


def test_read_write_index(tmpdir):
    index_filename = (tmpdir / "index").strpath
    nn_model = common.FaissNN()
    features = np.arange(3 * 6, dtype=np.float32).reshape(3, 6)
    nn_model.fit(features)
    nn_model.save(index_filename)

    loaded_nn_model = common.FaissNN()
    loaded_nn_model.load(index_filename)

    query_features = np.arange(10 * 6, dtype=np.float32).reshape(10, 6)
    assert loaded_nn_model.run(2, query_features) is not None
    assert np.all(
        loaded_nn_model.run(2, query_features)[0] == nn_model.run(2, query_features)[0]
    )
    assert np.all(
        loaded_nn_model.run(2, query_features)[1] == nn_model.run(2, query_features)[1]
    )


def test_average_merger_shape():
    input_features = []
    input_features.append(np.arange(2 * 3 * 4 * 5).reshape([2, 3, 4, 5]))
    input_features.append(2 * np.arange(2 * 3 * 4 * 5).reshape([2, 4, 3, 5]))

    merger = common.AverageMerger()
    output_features = merger.merge([input_features[0]])
    assert np.all(output_features.shape == (2, 3))

    merger = common.AverageMerger()
    output_features = merger.merge(input_features)
    assert np.all(output_features.shape == (2, 7))


def test_average_merger_output():
    input_features = [np.ones([2, 3, 4, 5])]

    merger = common.AverageMerger()
    output_features = merger.merge(input_features)
    assert np.all(output_features == 1.0)


def test_concat_merger_shape():
    input_features = []
    input_features.append(np.arange(2 * 3 * 4 * 5).reshape([2, 3, 4, 5]))
    input_features.append(2 * np.arange(2 * 3 * 4 * 5).reshape([2, 4, 3, 5]))

    merger = common.ConcatMerger()
    output_features = merger.merge([input_features[0]])
    assert np.all(output_features.shape == (2, 3 * 4 * 5))

    merger = common.ConcatMerger()
    output_features = merger.merge(input_features)
    assert np.all(output_features.shape == (2, 3 * 4 * 5 + 4 * 3 * 5))


def test_concat_merger_output():
    input_features = []
    input_features.append(np.ones([2, 3, 4, 5]))
    input_features.append(2 * np.ones([2, 3, 4, 5]))

    merger = common.ConcatMerger()
    output_features = merger.merge([input_features[0]])
    assert np.all(output_features == 1.0)

    merger = common.ConcatMerger()
    output_features = merger.merge(input_features)
    assert np.all(output_features[:, : 3 * 4 * 5] == 1.0)
    assert np.all(output_features[:, 3 * 4 * 5 :] == 2.0)
