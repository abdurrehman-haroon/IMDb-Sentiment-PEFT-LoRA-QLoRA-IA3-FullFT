# IMDb-Sentiment-PEFT-LoRA-QLoRA-IA3-FullFT

### *Parameter‑Efficient Fine‑Tuning Benchmarks on IMDb*

---

<div align="center">
  <b>Course :</b> L21‑5691 Generative AI (Spring 2025) 
  <b>Student :</b> Abdurrehman Haroon (21L‑5691)  •  <b>License:</b> MIT
</div>

---

## Project Overview

The goal is to compare **parameter‑efficient fine‑tuning (PEFT)** approaches against traditional full fine‑tuning on the **IMDb movie‑review sentiment** dataset.  We benchmark four strategies on `roberta‑base` (124 M params):

| Method      | Extra Trainable Params | GPU VRAM (GB) | Train Time (min) | Test Accuracy |
| ----------- | ---------------------: | ------------: | ---------------: | ------------: |
| **Full FT** |          124 M (100 %) |          2.64 |             10.2 |    **91.4 %** |
| **LoRA**    |        1.18 M (0.95 %) |          1.48 |          **7.6** |        91.2 % |
| **QLoRA**   |        1.03 M (0.83 %) |      **1.36** |             11.2 |        90.5 % |
| **IA³**     |    **0.66 M (0.53 %)** |          2.85 |              7.9 |        84.9 % |

> *LoRA reproduces 99.7 % of the full fine‑tune accuracy while updating <1 % of the weights and halving memory.*

---

## Key Highlights

* **Four PEFT recipes out‑of‑the‑box** (`peft` 0.15 API).
* **4‑bit QLoRA** pipeline – fits on consumer 4 GB GPUs.
* **Unified Trainer script** toggled via `--method {full,lora,qlora,ia3}`.
* **Live Matplotlib dashboards** for loss & metric curves.

---

## Dataset

| Split | Samples | Sentiment Classes |
| ----- | ------: | :---------------: |
| Train |   3 000 |    2 (pos / neg)  |
| Test  |   2 000 |          2        |

We subsample IMDb for quick classroom experiments; modify `--n_train` / `--n_test` for full 50 k.

Data are tokenised with **`roberta‑base`** max‑len 256.

---

## Fine‑Tuning Methods

### 1. Full Fine‑Tuning

All weights updated.  Highest capacity → highest compute cost.

### 2. LoRA

Low‑rank adapters *(r = 16, α = 32, dropout = 0.1)* injected into **Query & Value** of every attention layer; backbone frozen.

### 3. QLoRA

* Backbone quantised to **4‑bit NF4**.
* LoRA adapters *(r = 8)* trained on top.
* Double‑quant and paged optimisers via `bitsandbytes`.

### 4. IA³ Adapters

Per‑head **input / attention / output gain** vectors (≈0.5 % params).  Fastest & lightest, but lower accuracy.

---

## Experimental Setup

* **Model:** `roberta‑base` (124.6 M)
* **Hardware:** NVIDIA RTX 3050 (4 GB) + CUDA 12.3
* **Common hyper‑params:**

  ```yaml
  epochs: 3
  batch_size: 8
  lr: 2e-5
  max_length: 256
  seed: 42
  ```
* **Frameworks:** PyTorch 2.3, Hugging Face Transformers 4.38, PEFT 0.15, BitsAndBytes 0.41.

---

**Observations**

* *Accuracy:* Full FT > LoRA ≈ QLoRA ≫ IA³.
* *Memory:* QLoRA uses the least VRAM (1.36 GB).
* *Speed:* LoRA trains fastest; QLoRA slowed by 4‑bit kernels.
* *Params:* IA³ is the most frugal (0.53 %).


---

## Getting Started

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Troubleshooting & FAQ

| Issue              | Possible Fix                                                                    |
| ------------------ | ------------------------------------------------------------------------------- |
| CUDA out‑of‑memory | Reduce `batch_size`; switch to QLoRA.                                           |
| QLoRA very slow    | Ensure **bitsandbytes** compiled for your GPU; add `--flash_attn` if supported. |
| IA³ poor accuracy  | Increase epochs or merge with LoRA for hybrid tuning.                           |

---

## Contributing

Pull requests are welcome!  Please include unit tests (`pytest`) and follow **PEP‑8**.

---

## License

Released under the MIT License.  See `LICENSE` for details.


