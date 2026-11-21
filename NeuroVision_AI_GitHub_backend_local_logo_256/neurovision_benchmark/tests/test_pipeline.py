import numpy as np
import torch
from nv_benchmark.features import csi_to_features,standardize_fit
from nv_benchmark.models import PerceptionClassifier
from train import grouped_split


def test_csi_feature_shape_and_determinism():
    rng=np.random.default_rng(12)
    raw=rng.random((220,3,2,30)).astype('float32')
    a=csi_to_features(raw)
    assert a.shape==(12,128)
    assert np.array_equal(a,csi_to_features(raw))
    assert np.isfinite(a).all()


def test_group_split_no_overlap():
    groups=np.asarray([f'g{i//3}' for i in range(60)])
    a,b,c=grouped_split(groups)
    assert not(set(groups[a])&set(groups[b]))
    assert not(set(groups[a])&set(groups[c]))
    assert not(set(groups[b])&set(groups[c]))


def test_model_forward_three_modes():
    csi=torch.randn(4,12,128); camera=torch.randn(4,8)
    for mode in ('csi','camera','fusion'):
        model=PerceptionClassifier(mode)
        assert model(csi,camera).shape==(4,)


def test_train_only_normalization():
    x=np.ones((4,12,128),dtype='float32')
    x[2:]=20
    mu,sd=standardize_fit(x[:2])
    assert np.allclose(mu,1)
    assert np.all(sd>0)
