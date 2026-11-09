"""Local, multi-audio Qwen2.5-Omni adapter for context_pilot.py; imported only for explicit inference.

Uses the existing Thinker-only builder. No API, downloads, hidden-state extraction or audio concatenation.
"""
import copy
import time


class OmniContextPredictor:
    def __init__(self, model_dir, device, protocol):
        import torch
        import qwen2_5_omni
        from audio_llm import dtype_for, label_tokens, plain_stop_check_on_mps
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            if device == "cpu":
                raise RuntimeError("No CUDA/MPS device; pass --device cpu explicitly if intended")
        self.device, self.dtype = torch.device(device), dtype_for(device)
        if self.device.type == "mps":
            plain_stop_check_on_mps()
        self.model, self.processor = qwen2_5_omni.build(model_dir, self.device, self.dtype)
        self.labels = protocol["labels"]
        self.settings = protocol["config"]["decoding"]
        self.ids = label_tokens(self.processor.tokenizer, self.labels) if self.settings["save_label_probs"] else {}
        self.generation = copy.deepcopy(self.model.generation_config)
        self.generation.update(do_sample=False, temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0,
                               max_new_tokens=self.settings["max_new_tokens"], return_dict_in_generate=True,
                               output_logits=True, output_hidden_states=False)
        stops = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in qwen2_5_omni.STOP_TOKENS]
        known = self.generation.eos_token_id
        known = [] if known is None else [known] if isinstance(known, int) else list(known)
        self.generation.eos_token_id = list(dict.fromkeys(known + stops))
        if self.generation.pad_token_id is None:
            self.generation.pad_token_id = stops[-1]
        self.root = model_dir.resolve().parents[1]

    def __call__(self, messages):
        import numpy as np
        import torch
        from audio_io import load_audio, SR
        from parsing import parse_label
        messages = copy.deepcopy(messages)
        paths = [item["audio"] for message in messages for item in message["content"] if item["type"] == "audio"]
        # Same complete target in all conditions; each history/target has its own audio placeholder.
        waveforms = [load_audio(self.root / path) for path in paths]
        started = time.perf_counter()
        answers, found = [], {}
        for attempt in range(self.settings["retries_invalid"] + 1):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if text.count(self.processor.audio_token) != len(waveforms):
                raise RuntimeError("Audio placeholders and supplied waveforms do not match")
            inputs = self.processor(text=text, audio=waveforms, sampling_rate=SR, return_tensors="pt",
                                    audio_kwargs={"truncation": False})
            inputs = {k: v.to(self.device, self.dtype) if torch.is_tensor(v) and v.is_floating_point()
                      else v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            with torch.inference_mode():
                result = self.model.generate(**inputs, generation_config=self.generation)
                first = result.logits[0][0]
                if not bool(torch.isfinite(first).all()):
                    raise RuntimeError("Non-finite logits; stop and resume in a fresh process")
                if attempt == 0 and self.ids:
                    probabilities = torch.softmax(first.float(), dim=-1)
                    mass = {lab: float(probabilities[ids].sum()) for lab, ids in self.ids.items()}
                    total = sum(mass.values())
                    if not np.isfinite(total) or total <= 0:
                        raise RuntimeError("Invalid first-token label probability mass")
                    found = {"label_probs": {lab: value / total for lab, value in mass.items()}, "label_mass": total}
            generated = result.sequences[0, inputs["input_ids"].shape[1]:]
            answer = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
            answers.append(answer)
            label = parse_label(answer, self.labels)
            if label:
                break
            messages += [{"role": "assistant", "content": [{"type": "text", "text": answer}]},
                         {"role": "user", "content": [{"type": "text", "text":
                          "Answer with exactly one word from this list, in lowercase, and nothing else: "
                          + ", ".join(self.labels) + "."}]}]
        return {"prediction": label or "invalid", "answers": answers, "requests": len(answers), **found,
                "seconds": round(time.perf_counter() - started, 3), "device": str(self.device),
                "dtype": str(self.dtype).replace("torch.", ""),
                "audio_count": len(paths), "input_seconds": [len(wav) / SR for wav in waveforms]}
