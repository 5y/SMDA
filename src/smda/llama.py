"""Hugging Face Llama adapter with explicit token and layer conventions."""
from contextlib import contextmanager
import numpy as np
import torch


def prompt_tokens(tokenizer, prompt, max_tokens=None):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    if not ids:
        raise ValueError("Empty formatted prompt.")
    if max_tokens is not None and len(ids) > max_tokens:
        raise ValueError(f"Prompt has {len(ids)} tokens; limit is {max_tokens}. Increase the limit explicitly; no silent truncation.")
    return ids


def pad_right(sequences, pad_id, device):
    if not sequences or any(not s for s in sequences):
        raise ValueError("Cannot pad empty sequences.")
    ids = torch.full((len(sequences), max(map(len, sequences))), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(sequences):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, :len(s)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def refusal_logprobs(model, tokenizer, prompts, target_sequences, *, batch_size=8, max_tokens=None):
    """Sum joint probabilities of distinct, equal-length token sequences.

    Each suffix is appended BEFORE padding and scored under its own history.
    Equal-length distinct sequences are disjoint events; unequal/prefix-overlap
    sequences are rejected to avoid double counting probability mass.
    """
    targets = [list(map(int, s)) for s in target_sequences]
    if not targets or not targets[0] or len(set(map(tuple, targets))) != len(targets) or len({len(s) for s in targets}) != 1:
        raise ValueError("Targets must be unique, nonempty, equal-length sequences.")
    if batch_size < 1 or not prompts:
        raise ValueError("Need prompts and a positive batch size.")
    device = next(model.parameters()).device
    values = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                bases = [prompt_tokens(tokenizer, p, max_tokens) for p in prompts[start:start+batch_size]]
                variant_scores = []
                for suffix in targets:
                    inputs = pad_right([p + suffix for p in bases], tokenizer.pad_token_id, device)
                    logits = model(**inputs, use_cache=False).logits
                    # Gather only relevant positions before log_softmax, reducing memory.
                    positions = torch.tensor([[len(p)-1+k for k in range(len(suffix))] for p in bases], device=device)
                    selected = logits[torch.arange(len(bases), device=device)[:, None], positions].float()
                    lp = selected.log_softmax(-1)
                    target = torch.tensor(suffix, device=device)[None, :, None].expand(len(bases), -1, -1)
                    variant_scores.append(lp.gather(-1, target).squeeze(-1).sum(-1))
                values.append(torch.logsumexp(torch.stack(variant_scores), dim=0).cpu())
    finally:
        model.train(was_training)
    return torch.cat(values).double().numpy()


def residual_activations(model, tokenizer, prompts, *, layer=10, batch_size=8, max_tokens=None):
    """Post-block residual at the last user <|eot_id|>, zero-based block index.

    A direct hook on model.model.layers[10] matches blocks.10.hook_resid_post.
    hidden_states[10] is NOT used: it corresponds to the previous block.
    """
    if not prompts or batch_size < 1 or not 0 <= layer < len(model.model.layers):
        raise ValueError("Invalid prompts, batch size or block index.")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer has no Llama end-of-turn token.")
    device = next(model.parameters()).device
    values = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(prompts), batch_size):
            bases = [prompt_tokens(tokenizer, p, max_tokens) for p in prompts[start:start+batch_size]]
            positions = []
            for ids in bases:
                where = [i for i, t in enumerate(ids) if t == eot_id]
                if not where:
                    raise ValueError("Formatted prompt lost its end-of-turn marker.")
                positions.append(where[-1])
            inputs = pad_right(bases, tokenizer.pad_token_id, device)
            captured = []

            def capture(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                index = torch.tensor(positions, device=hidden.device)
                captured.append(hidden[torch.arange(len(bases), device=hidden.device), index].detach().float().cpu())

            handle = model.model.layers[layer].register_forward_hook(capture)
            try:
                with torch.no_grad():
                    model.model(**inputs, use_cache=False)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("Expected one residual capture per forward.")
            values.append(captured[0])
    finally:
        model.train(was_training)
    return torch.cat(values).double().numpy()


def sft_encoding(tokenizer, prompt, response, *, max_length=1024, response_only=True):
    prefix = prompt_tokens(tokenizer, prompt)
    ids = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False)
    if ids[:len(prefix)] != prefix:
        raise ValueError("Chat template does not share the expected assistant prefix.")
    ids = ids[:max_length]
    if len(ids) <= len(prefix):
        raise ValueError("Truncation removed the complete assistant response.")
    labels = ids.copy()
    if response_only:
        labels[:len(prefix)] = [-100] * len(prefix)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


@contextmanager
def single_example_update(model, tokenizer, prompt, response, *, learning_rate=1e-4, max_length=512):
    """One unnormalized SGD step; exact CPU snapshots restored even on failure.

    No optimizer momentum, accumulation across pairs, gradient clipping, or LoRA.
    Snapshotting trainable parameters needs roughly one model copy in CPU RAM.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    device = next(model.parameters()).device
    encoded = sft_encoding(tokenizer, prompt, response, max_length=max_length, response_only=True)
    batch = {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in encoded.items()}
    was_training = model.training
    backup = []
    model.zero_grad(set_to_none=True)
    # eval disables stochastic layers but does not disable autograd.
    model.eval()
    try:
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite SFT loss.")
        loss.backward()
        grad_squared = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise ValueError("Non-finite gradient.")
                    backup.append((p, p.detach().cpu().clone()))
                    grad_squared += float(p.grad.double().square().sum().cpu())
                    p.add_(p.grad, alpha=-learning_rate)
        model.zero_grad(set_to_none=True)
        yield {"loss": float(loss.detach().cpu()), "gradient_norm": grad_squared ** 0.5}
    finally:
        with torch.no_grad():
            for p, original in backup:
                p.copy_(original.to(p.device))
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def load_model(model_id, revision, *, device="cuda", dtype="float32", token=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, token=token,
        torch_dtype=getattr(torch, dtype), use_safetensors=True).to(device)
    return model.eval(), tokenizer
