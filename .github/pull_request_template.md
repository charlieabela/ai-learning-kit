## What changes and why

<!-- One or two sentences: the problem, and what this pull request does about it. -->

## Effect on notebooks and students

<!-- Do notebooks already handed out still run? Do any need restamping (dev.notebooks stamp) and redistributing? Anything students must do (git pull, a new file on the VLE)? Write "none" if nothing changes for them. -->

## Checked

- [ ] `python -m pytest -q` passes
- [ ] `python -m ai_learning_kit.dev.validate` passes
- [ ] Notebooks affected by the change re-executed, outputs compared (say which)
- [ ] CHANGELOG entry and version (`python -m ai_learning_kit.dev.release X.Y.Z`), if this is a release
- [ ] No links to Drive, OneDrive or SharePoint, no personal data, no attribution lines

## Not verified

<!-- What could not be run or checked, and why. Write "nothing" if everything was checked. -->

## After merging

- [ ] Release `vX.Y.Z` on main (Releases › Draft a new release › tag `vX.Y.Z`, *Create new tag on publish*, target `main`)
- [ ] Any `data-N` release this change refers to (pre-release, not the latest)
- [ ] Re-run the checks of the Notebook Tutor pull request that bundles this version, merge it and release it
