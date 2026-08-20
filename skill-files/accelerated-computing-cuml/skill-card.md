## Description: <br>
Official NVIDIA-authored guidance for NVIDIA cuML GPU machine learning, scikit-learn acceleration with cuml.accel, RandomForest, KMeans, UMAP, HDBSCAN, regression, classification, clustering, PCA, multi-GPU training. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to GPU-accelerate machine learning workflows using NVIDIA cuML, including scikit-learn migration, cuml.accel zero-code-change acceleration, and multi-GPU training. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Algorithm Guide](references/algorithm-guide.md) <br>
- [cuml.accel Reference](references/cuml-accel.md) <br>
- [Multi-GPU Training](references/multi-gpu-training.md) <br>
- [Unique APIs](references/unique-apis.md) <br>
- [cuML GitHub Repository](https://github.com/rapidsai/cuml) <br>
- [cuML API Reference](https://docs.rapids.ai/api/cuml/stable/api/) <br>
- [cuml.accel Benchmarks](https://docs.rapids.ai/api/cuml/stable/cuml-accel/benchmarks/) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>
- `codex` <br>



## Evaluation Tasks: <br>
Evaluated against 15 tasks (14 positive skill-activation, 1 negative) via NVSkills-Eval external profile with 2 attempts per task and a 50% pass threshold. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 97% (-3%) | 90% (-10%) |
| Correctness | 8 | 91% (+11%) | 89% (+9%) |
| Discoverability | 8 | 74% (+18%) | 63% (+10%) |
| Effectiveness | 8 | 86% (+2%) | 84% (+3%) |
| Efficiency | 8 | 52% (+15%) | 40% (+4%) |

## Skill Version(s): <br>
9d42395 (source: git SHA, committed 2026-05-29) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
