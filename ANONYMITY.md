# Anonymity statement

This repository accompanies a NeurIPS 2026 submission and is provided **anonymously** for double-blind review. All explicit author identifiers, institutional affiliations, cluster paths, and personal email addresses have been removed from the code, configuration, documentation, and PDF metadata.

## What we have done to preserve anonymity

- The `LICENSE` file lists "Anonymous Authors (NeurIPS 2026 submission)" as the copyright holder.
- The `CITATION.cff` and `pyproject.toml` author fields read "Anonymous Authors".
- All source files have been swept for personal names, institutional emails, internal cluster hostnames (e.g. `jupyterhub.*`), absolute home-directory paths (`/home/<user>/...`), and SLURM `--account` lines. Cluster-specific defaults are replaced with relative paths and a `# adapt to your cluster` comment.
- The submission PDF in `paper/paper.pdf` is compiled with the NeurIPS template's anonymous toggle (no `\final` flag), and its metadata fields (`Author`, `Creator`, `Producer`) have been verified empty or generic.
- Internal experiment identifiers (`J####`, internal job IDs) have been replaced with descriptive phrases or removed.

## What a reviewer should *not* attempt to deanonymize

We respectfully ask that reviewers do not search Git history, public GitHub mirrors, or external resources to attempt to identify the authors. The anonymous hosting service (`anonymous.4open.science`) automatically strips Git metadata; any code shared outside that service has been similarly cleansed.

## Post-acceptance

Upon acceptance, this repository will be replaced with a non-anonymous version that preserves the same code structure and includes proper authorship metadata, citations, and acknowledgements.
