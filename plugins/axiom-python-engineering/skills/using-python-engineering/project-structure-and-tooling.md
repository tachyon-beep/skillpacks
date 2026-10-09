# Python project and packaging boundaries

Use the existing project layout and package/environment tools. A small change does not need a scaffold, tool migration, new CI system or wholesale configuration rewrite.

## Inspect before editing

Read build-system settings, package discovery, entry points, dependency groups/extras, lock policy and supported Python versions. Establish whether commands execute the installed package, an editable install or the source tree; they may import different code.

## Failure checks

- A passing checkout import does not prove the built wheel contains modules, package data or command entry points. Inspect/install the artifact in a clean environment when packaging changes.
- Keep runtime dependencies distinct from development/test tools and optional extras. Test the combinations actually advertised.
- A lock file and broad package metadata serve different consumers. Preserve the project's intended reproducibility and library compatibility policies.
- Relative paths and current-working-directory assumptions often fail in installed packages. Resolve resources through the package/runtime contract.
- Check namespace packages and generated sources before moving code or adding `__init__.py`.
- Lint/type/test configuration belongs in its established owner. Duplicate competing configuration can make local and CI results disagree.
- Avoid unrequested formatter churn and automatic dependency upgrades in a behavioral repair.
- Publishing metadata, license files and credentials need separate attention; never assume a successful build authorizes publication.

## Evidence

Run affected tool commands in the intended environment. For distribution changes, build and inspect artifacts and exercise the installed consumer path. State which Python/platform/extras combinations were checked and what remains untested.
