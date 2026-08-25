# LLM Data Processing Agentic Skills

Skills for Claude Code and Codex that teach GPU-accelerated LLM data processing using cuDF, cuML, and NeMo Curator.

## Install Skills

For Claude Code and Codex to use these skills, clone the repo and copy the skill files to the appropriate locations.

```bash
git clone https://github.com/AllisonDing/LLM-data-processing-agentic-skills.git
cd LLM-data-processing-agentic-skills
```

**Claude Code:**
```bash
mkdir -p ~/.claude/skills
cp -r skill-files/accelerated-computing-cudf ~/.claude/skills/
cp -r skill-files/accelerated-computing-cuml ~/.claude/skills/
cp -r skill-files/nemo-curator ~/.claude/skills/
cp -r skill-files/nemo-curator-ray ~/.claude/skills/
```

**Codex:**

> **Note:** Codex reads `AGENTS.md` instead of `SKILL.md`. Before copying, rename each skill's `SKILL.md` to `AGENTS.md`:
> ```bash
> for skill in skill-files/accelerated-computing-cudf skill-files/accelerated-computing-cuml skill-files/nemo-curator skill-files/nemo-curator-ray; do
>     cp $skill/SKILL.md $skill/AGENTS.md
> done
> ```

```bash
mkdir -p ~/.agents/skills
cp -r skill-files/accelerated-computing-cudf ~/.agents/skills/
cp -r skill-files/accelerated-computing-cuml ~/.agents/skills/
cp -r skill-files/nemo-curator ~/.agents/skills/
cp -r skill-files/nemo-curator-ray ~/.agents/skills/
```

## Run NeMo Curator

To run NeMo Curator with GPU support, build from source and launch the container.

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
docker build -t nemo-curator-from-source -f docker/Dockerfile .
docker run --gpus all -it \
    -v /home/allisond/Curator:/workspace/Curator \
    nemo-curator-from-source bash
```