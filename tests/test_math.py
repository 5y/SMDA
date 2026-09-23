import numpy as np
import pytest
from sklearn.linear_model import Ridge
from smda.policy import ridge_weights, best_threshold, fit_policy, RidgePolicy
from smda.influence import decompose_influence, linearized_features
from smda.data import feature_matrix, validate_pairs
from smda.curation import strict_bool, quantitative_mask
from smda.metrics import soft_metrics


def test_ridge_matches_sklearn():
    rng=np.random.default_rng(42)
    X=rng.normal(size=(60,7)); Y=rng.normal(size=60)
    np.testing.assert_allclose(ridge_weights(X,Y,5), Ridge(alpha=5,fit_intercept=False).fit(X,Y).coef_, atol=1e-12)


@pytest.mark.parametrize("which", ["x","y","both"])
def test_influence_matches_central_finite_difference(which):
    rng=np.random.default_rng(19)
    X=rng.normal(size=(30,5)); Y=rng.normal(size=30)
    dx=rng.normal(size=X.shape) if which!="y" else np.zeros_like(X)
    dy=rng.normal(size=Y.shape) if which!="x" else np.zeros_like(Y)
    eps=1e-5
    actual=decompose_influence(X,Y,dx,dy,alpha=5)
    numerical=(ridge_weights(X+eps*dx,Y+eps*dy,5)-ridge_weights(X-eps*dx,Y-eps*dy,5))/(2*eps)
    np.testing.assert_allclose(actual.delta_w,numerical,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(actual.delta_w, actual.delta_w_x+actual.delta_w_y)
    assert actual.score == pytest.approx(actual.delta_w @ ridge_weights(X,Y,5))


def test_relu_mask_and_derivative():
    h=np.array([[2.,-3.],[-2.,3.]])
    dh=np.ones_like(h); w=np.eye(2); b=np.array([0.,0.])
    dx=linearized_features(dh,h,w,b)
    assert np.array_equal(dx,[[1,0],[0,1]])
    eps=1e-6
    numerical=(np.maximum((h+eps*dh)@w+b,0)-np.maximum((h-eps*dh)@w+b,0))/(2*eps)
    np.testing.assert_allclose(dx,numerical)


def test_threshold_ties_and_extremes():
    assert best_threshold([1,1,2],[1,1,0]) == 1
    assert best_threshold([1,2],[0,0]) > 2
    assert best_threshold([1,2],[1,1]) == 1


def test_saved_policy_requires_same_feature_order(tmp_path):
    p=fit_policy([[1,2],[3,4]],[1,0],[9,2],labels=[1,0])
    p.save(tmp_path/'policy.json'); loaded=RidgePolicy.load(tmp_path/'policy.json')
    np.testing.assert_allclose(p.predict([[2,3]],[9,2]),loaded.predict([[2,3]],[9,2]))
    with pytest.raises(ValueError): loaded.predict([[2,3]],[2,9])


def test_features_use_numeric_ids_not_label_text_or_lexicographic_order():
    x,ids=feature_matrix([{'feat_10_new_label':2,'feat_2_old_label':5}])
    assert ids==[2,10] and x.tolist()==[[5,2]]
    with pytest.raises(ValueError): feature_matrix([{'feat_2_a':1,'feat_2_b':2}])


def test_overlap_and_duplicate_pairs_rejected():
    row=dict(prompt=' test ',response='answer',label='harmless_compliance')
    with pytest.raises(ValueError): validate_pairs([row],['test'])
    with pytest.raises(ValueError): validate_pairs([row,row])


def test_curation_not_python_string_truthiness():
    assert strict_bool('FALSE') is False
    with pytest.raises(ValueError): strict_bool('')
    mask,_=quantitative_mask([-4,2,-1,4],['harmful_refusal']*2+['harmless_compliance']*2)
    assert mask.tolist()==[True,False,False,True]


def test_soft_metrics_hand_calculation():
    result=soft_metrics([0.8,0.6,0.2,0.1],[1,1,0,0])
    assert result['correctness']==pytest.approx(0.775)
    assert result['recall']==pytest.approx(0.7)
    assert result['precision']==pytest.approx(1.4/1.7)


def test_feature_selection_and_contrastive_examples():
    from smda.features import select_features, labeling_examples
    X=np.array([[3,4,0],[2,0,0],[0,0,3],[0,0,2]])
    chosen=select_features(X,[1,1,0,0])
    assert [r['direction'] for r in chosen]==['harmful','harmful','harmless']
    examples=labeling_examples(X,0,[0,1],contrastive_n=1,extremes_n=1)
    assert examples['top']==[0] and examples['contrastive']==[1]
