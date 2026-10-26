# Release sanitization

This code-only IDARec snapshot was prepared for private storage and subsequent
anonymous review. The input archive was not modified.

- Replaced development-machine directories, personal path components and
  internal server addresses with generic local-resource paths.
- Removed the export-specific handover note and regenerated the export manifest.
- Applied the same sanitization to every content-addressed object, including
  historical backend variants and patch files; regenerated object names and
  both backend profiles.
- Added an ignored `resources.local.json` override; preflight reads the selected
  base-model link. Replaced machine-specific GPU lock and environment paths.
- Preserved algorithm code, training settings, input asset checksums, and
  upstream attribution. Internal score-mode identifiers are unchanged.
- Excluded credentials, environments, data, weights, and generated outputs from
  future Git tracking. Historical queue cleanup authorization remains null.

The supplied historical audit files describe the source snapshot. Current
packaging checks are recorded separately in `audit/release_validation.json`.
No new training run or result claim is made by this release.
