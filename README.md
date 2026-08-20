# LLM Data Processing — Agentic Skills

This repo provides GPU-accelerated data processing skills for AI coding agents (Claude Code, Codex) and a Docker environment with the required Python libraries pre-installed.

## Skills Included

| Skill | Description |
|---|---|
| `accelerated-computing-cudf` | NVIDIA cuDF — GPU DataFrames and pandas acceleration |
| `accelerated-computing-cuml` | NVIDIA cuML — GPU-accelerated machine learning |
| `nemo-curator` | NVIDIA NeMo Curator — scalable multimodal dataset preparation |
| `nemo-curator-ray` | NeMo Curator Ray Data backend — distributed pipeline tuning |

## Prerequisites

- NVIDIA GPU with driver 580+ (CUDA 13)
- Docker with NVIDIA Container Toolkit
- Claude Code (`claude`) and/or Codex (`codex`) installed

## 1. Clone the Repo

    git clone https://github.com/AllisonDing/LLM-data-processing-agentic-skill.git
    cd LLM-data-processing-agentic-skill

## 2. Deploy Skills to Claude Code and Codex

    # Claude Code
    mkdir -p ~/.claude/skills
    cp -r skill-files/accelerated-computing-cudf ~/.claude/skills/
    cp -r skill-files/accelerated-computing-cuml ~/.claude/skills/
    cp -r skill-files/nemo-curator ~/.claude/skills/
    cp -r skill-files/nemo-curator-ray ~/.claude/skills/

    # Codex
    mkdir -p ~/.agents/skills
    cp -r skill-files/accelerated-computing-cudf ~/.agents/skills/
    cp -r skill-files/accelerated-computing-cuml ~/.agents/skills/
    cp -r skill-files/nemo-curator ~/.agents/skills/
    cp -r skill-files/nemo-curator-ray ~/.agents/skills/

## 3. Build the Docker Environment (Optional)

    docker build -t llm-data-processing .
    docker run --gpus all -it llm-data-processing bash

Test inside the container:

    python -c "import cudf, cuml, nemo_curator; print('All OK')"

## How It Works

    Claude Code / Codex          Docker Container
    (reads skills, generates) -> (executes GPU code)
         ~/.claude/skills/           cuDF + cuML
         ~/.agents/skills/           NeMo Curator
