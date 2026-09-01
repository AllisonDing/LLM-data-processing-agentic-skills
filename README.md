# LLM Data Processing Agentic Skills

Skills for Claude Code and Codex that teach GPU-accelerated LLM data processing using cuDF, cuML, and NeMo Curator.

## Install Skills

For Claude Code and Codex to use these skills, clone each skill's source repo and copy the skill files to the appropriate locations.

**Claude Code:**
```bash
mkdir -p ~/.claude/skills

# 1. cuDF skill
git clone https://github.com/NVIDIA/skills.git
cp -r skills/skills/accelerated-computing-cudf ~/.claude/skills/

# 2. cuML skill (internal GitLab — NVIDIA network required)
git clone https://gitlab-master.nvidia.com/RAPIDS/rapids-agent-skills.git
cp -r rapids-agent-skills/skills/accelerated-computing-cuml ~/.claude/skills/

# 3. NeMo Curator + Ray Data skills
git clone https://github.com/NVIDIA-NeMo/Curator.git
mkdir -p ~/.claude/skills/nemo-curator
cp Curator/AGENTS.md ~/.claude/skills/nemo-curator/SKILL.md

mkdir -p ~/.claude/skills/nemo-curator-ray
cp Curator/nemo_curator/backends/ray_data/AGENTS.md ~/.claude/skills/nemo-curator-ray/SKILL.md
```

**Codex:**

> **Note:** Codex reads `AGENTS.md` instead of `SKILL.md`. When copying the cuDF and cuML skills, rename `SKILL.md` to `AGENTS.md`; the NeMo Curator skills already ship as `AGENTS.md`, so no rename is needed.

```bash
mkdir -p ~/.agents/skills

# 1. cuDF skill
git clone https://github.com/NVIDIA/skills.git
cp -r skills/skills/accelerated-computing-cudf ~/.agents/skills/
mv ~/.agents/skills/accelerated-computing-cudf/SKILL.md ~/.agents/skills/accelerated-computing-cudf/AGENTS.md

# 2. cuML skill (internal GitLab — NVIDIA network required)
git clone https://gitlab-master.nvidia.com/RAPIDS/rapids-agent-skills.git
cp -r rapids-agent-skills/skills/accelerated-computing-cuml ~/.agents/skills/
mv ~/.agents/skills/accelerated-computing-cuml/SKILL.md ~/.agents/skills/accelerated-computing-cuml/AGENTS.md

# 3. NeMo Curator + Ray Data skills
git clone https://github.com/NVIDIA-NeMo/Curator.git
mkdir -p ~/.agents/skills/nemo-curator
cp Curator/AGENTS.md ~/.agents/skills/nemo-curator/AGENTS.md

mkdir -p ~/.agents/skills/nemo-curator-ray
cp Curator/nemo_curator/backends/ray_data/AGENTS.md ~/.agents/skills/nemo-curator-ray/AGENTS.md
```

## Run NeMo Curator

To run NeMo Curator with GPU support, build from source and launch the container.

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
docker build -t nemo-curator-from-source -f docker/Dockerfile .
docker run --gpus all -it \
    -v $HOME/Curator:/workspace/Curator \
    nemo-curator-from-source bash
```
