# DEV article workflow

The article is intentionally a draft. DEV is the original publication for now,
so it has no artificial canonical URL.

## Review and validate

```bash
node blog/publish.mjs jpeg-ai-on-apple-silicon --dry-run
```

Before publishing, replace the one-image smoke result with an uncontended,
multi-image run using warmups and repetitions, and resolve or clearly retain
the known Core ML bottom-edge difference.

Regenerate the measured images with:

```bash
uv run python blog/render_assets.py \
  runs/20260909T1340Z-timing-split/results.csv \
  runs/20260909T1340Z-timing-split/artifacts \
  /path/to/encode.visualizations blog/assets
```

## Create or update a private DEV draft

Push the image assets first so their raw GitHub URLs exist, then:

```bash
cp blog/.env.example blog/.env
# Add the DEV API key to blog/.env; that file is ignored by Git.
node blog/publish.mjs jpeg-ai-on-apple-silicon
```

The script creates a draft or updates the existing draft with the same title.
It cannot publish and refuses to modify an already-public article. Review the
DEV renderer and publish manually. The X post is deliberately deferred until
the corpus benchmark and native codec are mature.
