# Reward Modeling and Direct Preference Optimization (DPO)

This project explores **reward model training**, **DPO fine-tuning**, and **text-to-text DPO** for aligning language models with human preferences. All experiments use [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) as the base model and [PKU-Alignment/align-anything](https://github.com/PKU-Alignment/align-anything) as the training framework.

## Table of Contents

- [Environment Setup](#environment-setup)
- [1. Reward Model](#1-reward-model)
  - [1.1 Preference Dataset Template](#11-preference-dataset-template)
  - [1.2 Training](#12-training)
  - [1.3 Evaluation](#13-evaluation)
  - [1.4 Visualization](#14-visualization)
  - [1.5 Discussion](#15-discussion)
- [2. DPO Fine-Tuning](#2-dpo-fine-tuning)
  - [2.1 Training](#21-training)
  - [2.2 Evaluation & Case Analysis](#22-evaluation--case-analysis)
  - [2.3 Discussion](#23-discussion)
- [3. (Bonus) Text-to-Text DPO Notebook](#3-bonus-text-to-text-dpo-notebook)
- [Repository Structure](#repository-structure)
- [References](#references)

## Environment Setup

Install the [align-anything](https://github.com/PKU-Alignment/align-anything) package:

```bash
git clone git@github.com:PKU-Alignment/align-anything.git
cd align-anything
conda create -n align-anything python==3.11
conda activate align-anything

# (Optional) Install CUDA in conda
conda install nvidia/label/cuda-12.2.0::cuda
export CUDA_HOME=$CONDA_PREFIX

# Install align-anything
pip install -e .[all]
```

## 1. Reward Model

### 1.1 Preference Dataset Template

A custom `HOMEWORK` template was implemented to convert the [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) preference dataset into the align-anything format. The template maps `question` to the user prompt, and uses `overall_response` to determine which of the two responses is preferred (better) vs. rejected (worse):

```python
@register_template('HOMEWORK')
class HOMEWORK(BaseFormatter):
    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['question']
        better_response = raw_sample[f"response_{int(raw_sample['overall_response'])}"]
        worse_response = raw_sample[f"response_{3 - int(raw_sample['overall_response'])}"]
        better_conversation = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': better_response},
        ]
        worse_conversation = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': worse_response},
        ]
        meta_info = {
            'better_response': better_response,
            'worse_response': worse_response,
        }
        return better_conversation, worse_conversation, meta_info
```

### 1.2 Training

The reward model was trained for 3 epochs using DeepSpeed with default hyperparameters. Evaluation was performed concurrently during training to monitor quality.

```bash
deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.text_to_text.rm \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_template HOMEWORK \
    --train_datasets ${TRAIN_DATASETS} \
    --train_split train \
    --eval_datasets ${EVAL_DATASETS} \
    --eval_template HOMEWORK \
    --eval_split validation \
    --eval_strategy steps \
    --output_dir ${OUTPUT_DIR} \
    --epochs 3
```

**Training results:** Loss decreased steadily over training steps. Validation accuracy showed no downward trend, indicating no significant overfitting. The accuracy plateaued in the second half, suggesting ~1 epoch may suffice for convergence.

### 1.3 Evaluation

The `eval` function in `align_anything/trainers/text_to_text/rm.py` was modified to save all scored texts (both chosen and rejected, with their reward scores) for downstream visualization and analysis.

### 1.4 Visualization

Two visualizations are provided in the `visualization/` directory:

- **`delta_hist.png`** -- Distribution of reward score differences (chosen - rejected). The distribution skews right, indicating that in most cases the chosen response receives a higher reward than the rejected one, confirming alignment with human preferences.

- **`score_hist.png`** -- Overlaid reward score distributions for chosen vs. rejected responses. Chosen responses have overall higher scores, demonstrating good discriminative ability. Some overlap exists, likely due to inherently ambiguous samples or model limitations.

### 1.5 Discussion

**Applications of reward modeling:** Reward models have broad applications including dialogue (improving accuracy, emotional expression), reasoning (mathematical logic, code generation efficiency), and retrieval & recommendation (relevance assessment, adaptive retrieval strategies using boolean vs. dense retrieval depending on query specificity).

**Robustness challenges:**
1. **Reward hacking** -- The policy model exploits loopholes rather than truly learning preferred content, due to lack of fine-grained reward signals.
2. **Reward model bias** -- The policy learns spurious correlations from training data (e.g., longer = better, more polite = better), deviating from actual preferences.
3. **Overfitting** -- The reward model memorizes annotator preferences or specific samples instead of building a generalizable preference function.
4. **Over-optimization & distribution shift** -- Aggressive RL optimization causes the policy output to drift from the reward model's training distribution, leading to high scores but low quality.

**Mitigating length bias:** Approaches include dual-head architectures like ODIN that separate content quality from length signals, explicit length bias fitting and removal (FiMi-RM), and KL regularization during the optimization phase to curb verbose generation.

**Modeling diverse human preferences:** Methods include Bayesian reward modeling to capture preference distributions rather than simple averages, synthetic preference augmentation with margin loss regularization, reward model ensembles with latent-space constraints, safety-oriented frameworks like COBRA for filtering malicious feedback, and multi-objective decomposition approaches like ArmoRM.

## 2. DPO Fine-Tuning

### 2.1 Training

DPO fine-tuning was performed for 1 epoch using the same `HOMEWORK` template:

```bash
deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.text_to_text.dpo \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_template HOMEWORK \
    --train_datasets ${TRAIN_DATASETS} \
    --train_split train \
    --output_dir ${OUTPUT_DIR} \
    --epochs 1
```

### 2.2 Evaluation & Case Analysis

Both the base model and DPO-tuned model were evaluated on the test set. Responses were generated and scored by the reward model. Key findings:

**Score comparison:** The DPO model achieves overall higher scores with a more concentrated distribution compared to the base model (see `visualization/dpo_comparison_results.csv`).

**Where DPO improved the model:**

| Improvement | Example |
|---|---|
| **Refusing illegal/harmful requests** | When asked how to steal a car, the DPO model refuses and advises against it, while the base model provides suggestions. |
| **Better emotional intelligence** | When faced with "I hate men!", the DPO model responds with empathy first then gently encourages inclusivity, rather than directly criticizing the user. |
| **Copyright awareness** | When asked to download content for free, the DPO model flags the issue and suggests legal alternatives instead of providing piracy links. |
| **More comprehensive answers** | For sensitive scenarios, the DPO model provides fuller, more nuanced responses with appropriate caveats. |

**Where DPO degraded performance:** On certain tasks (e.g., translation), DPO over-optimized for alignment signals at the cost of linguistic coherence, producing abnormally long outputs with severe repetition (see `visualization/worst_cases.csv`).

### 2.3 Discussion

**On-policy vs. off-policy:** DPO is an **off-policy, offline** method. It trains directly on a pre-collected static preference dataset, bypassing the online sampling loop used by PPO-style RLHF.

**Key insight of DPO over traditional RLHF:** DPO reformulates the RLHF objective as a supervised learning problem, directly optimizing the policy on preference data without requiring an explicit reward model or complex RL procedure. This improves stability, efficiency, and reduces implementation complexity.

**Limitations of DPO vs. RLHF:**
- Implicit reward may introduce bias
- Purely offline training limits generalization to unseen distributions
- Potential performance degradation on tasks not covered by preference data
- Objective design mismatch in certain scenarios

**Recent improvements on DPO:**

| Method | Key Innovation |
|---|---|
| **KTO** | Uses Kahneman-Tversky prospect theory utility functions; requires only binary (good/bad) feedback instead of pairwise comparisons |
| **SimPO** | Uses average log-probability as implicit reward (no reference model needed); introduces preference margin mechanism; normalizes for response length |
| **ORPO** | Single-stage method that integrates preference alignment directly into SFT via log-odds penalty; ~50% less compute than two-stage approaches |
| **IPO** | Adds identity mapping regularization to DPO loss to mitigate overfitting |
| **CPO** | Omits reference model and uses contrastive loss for comparable alignment with less memory |

## 3. (Bonus) Text-to-Text DPO Notebook

A self-contained Jupyter notebook (`text_to_text_dpo.ipynb`) demonstrates the full DPO training pipeline:

1. **Baseline evaluation** -- Load Qwen2.5-0.5B-Instruct and test zero-shot generation
2. **DPO training** -- Implement the DPO loss function and train with AdamW optimizer:
   ```
   L_DPO = -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected)))
   ```
   where `beta = 0.1`, using a frozen copy of the base model as the reference.
3. **Post-training evaluation** -- Compare model outputs before and after DPO training

The notebook uses the `align-anything` `PreferenceDataset` with the `HOMEWORK` template and trains for 100 steps with batch size 2.

## Repository Structure

```
.
├── README.md                          # This file
├── hw2.pdf                            # Full report (in Chinese)
├── text_to_text_dpo.ipynb             # Bonus: self-contained DPO training notebook
└── visualization/
    ├── delta_hist.png                 # Reward gap distribution (chosen - rejected)
    ├── score_hist.png                 # Reward score distributions for chosen vs. rejected
    ├── dpo_comparison_results.csv     # DPO vs. base model generation examples with scores
    └── worst_cases.csv                # Cases where the reward model performed worst
```

## References

- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) -- Rafailov et al., 2023
- [PKU-Alignment/align-anything](https://github.com/PKU-Alignment/align-anything)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) -- Ethayarajh et al., 2024
- [SimPO: Simple Preference Optimization](https://arxiv.org/abs/2405.14734) -- Meng et al., 2024
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) -- Hong et al., 2024
- [ODIN: Disentangled Reward Mitigates Hacking in RLHF](https://arxiv.org/abs/2402.07319) -- Chen et al., 2024
