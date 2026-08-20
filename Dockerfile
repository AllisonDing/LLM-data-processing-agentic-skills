# Base: RAPIDS 26.04 — includes cuDF, cuML, cuGraph via conda
FROM nvcr.io/nvidia/rapidsai/base:26.04-cuda13-py3.12

# Install NeMo Curator (cuda + ray extras)
RUN pip install uv && \
    uv pip install --system "nemo-curator[cuda,ray]" \
    --extra-index-url https://pypi.nvidia.com

# Copy skill files into the container
COPY skill-files/accelerated-computing-cudf  /opt/skills/accelerated-computing-cudf
COPY skill-files/accelerated-computing-cuml  /opt/skills/accelerated-computing-cuml
COPY skill-files/nemo-curator/AGENTS.md      /opt/skills/nemo-curator/AGENTS.md
COPY skill-files/nemo-curator-ray/AGENTS.md  /opt/skills/nemo-curator-ray/AGENTS.md

ENV SKILLS_DIR=/opt/skills

# WORKDIR must come before RUN mkdir so /workspace exists
WORKDIR /workspace

# Wire skills into Claude Code's expected locations
RUN mkdir -p .claude/skills && \
    ln -s /opt/skills/accelerated-computing-cudf .claude/skills/accelerated-computing-cudf && \
    ln -s /opt/skills/accelerated-computing-cuml .claude/skills/accelerated-computing-cuml && \
    ln -s /opt/skills/nemo-curator /workspace/.claude/skills/nemo-curator && \
    ln -s /opt/skills/nemo-curator-ray /workspace/.claude/skills/nemo-curator-ray && \
    cp /opt/skills/nemo-curator/AGENTS.md AGENTS.md

# Wire skills into Codex's expected locations
RUN mkdir -p $HOME/.agents/skills && \
    ln -s /opt/skills/accelerated-computing-cudf $HOME/.agents/skills/accelerated-computing-cudf && \
    ln -s /opt/skills/accelerated-computing-cuml $HOME/.agents/skills/accelerated-computing-cuml && \
    ln -s /opt/skills/nemo-curator $HOME/.agents/skills/nemo-curator && \
    ln -s /opt/skills/nemo-curator-ray $HOME/.agents/skills/nemo-curator-ray
