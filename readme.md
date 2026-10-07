# Low-Rank Hierarchical Merging for Efficient Long-to-Short Reasoning

**Zeqiu Yu**, **Xuesheng Zhang**, **Wenxiao Zhao**, **Shixiao Wang**


---

## Introduction

Long-to-Short (L2S) model merging aims to combine the strong reasoning  
capability of a long chain-of-thought model with the concise style of a base  
model, but existing methods based on linear interpolation and first-order  
activation statistics ignore how the loss landscape responds to parameter  
displacement, which degrades accuracy on hard tasks. We propose **Low-Rank  
Hierarchical Merging (LHM)**, a training-free framework that replaces  
heuristic statistics with lightweight second-order curvature information: a  
data-free low-rank Hessian trace estimator (Hutchinson's estimator driven by  
random token sequences), a Layer Second-Order Sensitivity (LSOS) index that  
couples curvature with task-vector norm, and a Dual-Constraint Coefficient  
Mapping (DCCM) that turns LSOS scores into layer-adaptive merge coefficients  
under a fusion-error bound and a global length-reduction target. On six  
mathematical benchmarks with 1.5B and 7B Qwen2.5-Math / R1-Distill pairs, LHM  
outperforms state-of-the-art baselines such as ACM-TIES and ORCA, gaining over  
4% accuracy while cutting output length by up to 93.1%.

---

## Environment

```bash
# core merging environment
conda create -n lhm python=3.10 -y
conda activate lhm
pip install -r requirements.txt          # numpy==1.26.4, torch==2.5.1, transformers==4.48.2

# CPU-only is enough for the smoke test / unit tests
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python scripts/smoke_test.py
python tests/run_tests.py
```

```bash
# evaluation environment (Qwen2.5-Math toolkit)
git clone https://github.com/QwenLM/Qwen2.5-Math.git
pip install -r requirements-eval.txt
pip install -e Qwen2.5-Math/evaluation/latex2sympy

cp evaluation/l2s_eval.sh     Qwen2.5-Math/evaluation/
cp evaluation/data_process.py Qwen2.5-Math/evaluation/
```

---

## Running

All commands are run from `Low-Rank-Hierarchical-Merging/`.

### Comparison methods

```bash
# 1.5B pair (Qwen2.5-Math-1.5B + DeepSeek-R1-Distill-Qwen-1.5B)
for B in model_soup ta ties dare aim acm_ties orca_aim; do
  python scripts/merge_baselines.py --scale 1.5b --baseline $B \
      --device cuda --dtype bfloat16 \
      --output_model_dir outputs/1p5b/models/${B}
done
```

### Our method (LHM)

```bash
# 1) one-time, data-free Hessian trace estimation (Phase 2)
python scripts/extract_hessian.py --scale 1.5b --hessian_source base \
    --rank 16 --num_sequences 32 --seq_len 512 \
    --device cuda --dtype bfloat16 \
    --output outputs/1p5b/traces/base_k16.json

# 2) build LHM-TA / LHM-TIES / LHM-AIM
for M in ta ties aim; do
  python scripts/build_lhm_model.py --scale 1.5b --method $M \
      --rank 16 --alpha 0.15 --lambda_bar 0.5 \
      --trace_cache outputs/1p5b/traces/base_k16.json \
      --output_dir outputs/1p5b/lhm_${M} \
      --output_model_dir outputs/1p5b/models/LHM-${M} \
      --device cuda --dtype bfloat16
done
```

For the 7B pair, replace `--scale 1.5b` with `--scale 7b` and use  
`--method ta` / `--method aim`.

---

## Citation

```bibtex
@inproceedings{lhm2026lowrank,
  title     = {Low-Rank Hierarchical Merging for Efficient Long-to-Short Reasoning},
  author    = {Zeqiu Yu and Xuesheng Zhang and Wenxiao Zhao and Shixiao Wang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
