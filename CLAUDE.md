# ViralCut AI: notes for Claude

## Where the project lives

- The user's working copy is **`E:\Dev\ViralCutIA`** on Windows (it was moved from `C:`). Every
  change, addition, path in docs and instruction given to the user must target that folder.
- Nothing the app writes may land outside the project folder (in particular not in the Windows
  user profile on `C:`). `apps/api/viralcut/config.py` anchors everything to the checkout:
  - data, uploads, clips, database, transcript cache: `apps/api/data`
  - temp files: `apps/api/data/tmp` (`tempfile.tempdir`)
  - downloaded AI models: `models/` (`VIRALCUT_MODELS_DIR`, and `HF_HOME=models/hf`)
  - `.env`: `apps/api/.env`, found whatever the current directory is
  - pip/npm caches: `.cache/` (set in `scripts/atualizar-viralcut.bat`)
- New code that downloads or writes files must use these settings, never `~`, `%APPDATA%`,
  `%TEMP%` or a path relative to the current directory. Keep paths relative to the project so the
  folder can move again.
- The user runs ViralCut with `scripts\iniciar-viralcut.bat` (API with the embedded worker + Next.js
  production build) and updates with `scripts\atualizar-viralcut.bat`. The user-facing guide is
  `docs/COMECAR.md` (Portuguese).
