"""Small random Llama; checks tensor semantics without downloading any model."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
transformers=pytest.importorskip('transformers')
from smda.llama import refusal_logprobs, residual_activations, single_example_update, sft_encoding


class TinyTokenizer:
    pad_token_id=0
    unk_token_id=None
    def convert_tokens_to_ids(self, token): return 3 if token=='<|eot_id|>' else None
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids=[1]+[5+ord(c)%15 for c in messages[0]['content']]+[3,4]
        if len(messages)==2: ids += [5+ord(c)%15 for c in messages[1]['content']]+[3]
        return ids


@pytest.fixture
def tiny():
    torch.manual_seed(1)
    cfg=transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0)
    return transformers.LlamaForCausalLM(cfg).eval(), TinyTokenizer()


def test_logprob_invariant_to_batch_padding_and_real_prefix_history(tiny):
    model,tok=tiny; prompts=['a','longer']; targets=[[7,8,9],[10,11,12]]
    batched=refusal_logprobs(model,tok,prompts,targets,batch_size=2)
    separate=refusal_logprobs(model,tok,prompts,targets,batch_size=1)
    np.testing.assert_allclose(batched,separate,atol=2e-6)
    expected=[]
    for p in prompts:
        base=tok.apply_chat_template([{'role':'user','content':p}],add_generation_prompt=True)
        scores=[]
        for seq in targets:
            ids=torch.tensor([base+seq]); logits=model(ids).logits.log_softmax(-1)
            scores.append(sum(logits[0,len(base)-1+k,t] for k,t in enumerate(seq)))
        expected.append(torch.logsumexp(torch.stack(scores),0).item())
    np.testing.assert_allclose(batched,expected,atol=2e-6)


def test_residual_hook_correct_block_and_eot(tiny):
    model,tok=tiny
    got=residual_activations(model,tok,['abc','longer'],layer=1,batch_size=2)
    expected=[]
    for p in ['abc','longer']:
        ids=tok.apply_chat_template([{'role':'user','content':p}])
        out=model(torch.tensor([ids]),output_hidden_states=True)
        expected.append(out.hidden_states[2][0,len(ids)-2].detach().numpy())
    np.testing.assert_allclose(got,expected,atol=1e-7)
    assert not model.model.layers[1]._forward_hooks


@pytest.mark.parametrize('fail',[False,True])
def test_exact_restoration_after_update_and_exception(tiny,fail):
    model,tok=tiny
    before={k:v.clone() for k,v in model.state_dict().items()}
    try:
        with single_example_update(model,tok,'hello','response',learning_rate=0.1):
            assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
            if fail: raise RuntimeError('simulated evaluation failure')
    except RuntimeError:
        if not fail: raise
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert not model.training


def test_truncation_cannot_remove_entire_response(tiny):
    _,tok=tiny
    with pytest.raises(ValueError): sft_encoding(tok,'a long prompt','r',max_length=3)
