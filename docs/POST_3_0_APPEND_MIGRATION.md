# Post-3.0 Append migration candidate

This is an unreleased, isolated fix based on 04d732a. It does not change defaults, published tags, or the 3.0.0 qualification record.

File > Append now runs the existing resolution migration for newly imported local domain objects and dependent objects. Imported scenes also receive the existing removed-colour-mode migration. Linked library IDs and reused local IDs are not rewritten. Previously authored Resolution Divisions remain authoritative.

A library-load can create an unlinked domain whose world matrix is not evaluated. Such a domain is queued without stamping a guessed resolution; its first linked-scene dependency update applies the migration using evaluated parent transforms. Deleting that object, loading another file, or unregistering clears its pending record. This is not a general whole-file migration on every dependency update.

Acceptance: tests/acceptance/blender_append_migration_acceptance.py exercises actual Object/Scene Append, parent transforms, unlinked-library import and later scene evaluation, Link protection, existing-domain protection, idempotent registration, and save/reopen persistence in Blender 5.2. The script is wired into BACKGROUND_ACCEPTANCE. Handler ownership/persistence inventory includes the new callbacks. Unit coverage lives in tests/ui/test_append_migration.py alongside existing resolution/file-compatibility tests.

Evidence from the first real regression control: the same legacy fixture expected 229 divisions but current 3.0 loaded 64. The candidate preserves 229. Source tests alone do not qualify an installed extension or a release.
