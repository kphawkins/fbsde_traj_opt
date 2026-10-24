
Notes for code style
* Use trailing return syntax for all functions
* Use `noexcept` by default
* Put `const` on the left
* 120 character lines
* Represent all indices with `std::size_t`
* To enforce invariants of classes use a `Make` factory constructor pattern which returns a `Result`
  or an error if the invariants would not be satisfied by the inputs.

Research workflow
* Research happens on the `research` branch. `main` holds only code that has been reviewed, so never
  target `main` unless asked.
* For each change, branch off `research`, commit, push, open a PR into `research`, and merge it
  right away with a merge commit. No review and no CI: the CI workflow only runs for `main`.
* Commits can be coarse, but keep the conventional prefixes (`feat:`, `fix:`, `refactor:`, ...).
* Prefer extending an existing experiment binary with a flag over copying it into a new variant.
* Commit study results (CSV, logs, JSON, HTML reports and the scripts that analyze them) next to
  the experiment that produced them. Never commit binaries.
* `bazel test //...` skips tests tagged `slow`; `--config=slow` runs them too.

Cloud sessions (Claude Code on the web)
* `gh pr ...` and other GraphQL-backed `gh` commands are blocked in cloud sessions. Use the REST API:
  * Open: `gh api repos/kphawkins/fbsde_traj_opt/pulls -f base=research -f head=<branch> -f title=<title> -f body=<body>`
  * Merge: `gh api -X PUT repos/kphawkins/fbsde_traj_opt/pulls/<number>/merge -f merge_method=merge`
* If `clang++ --version` is not 21 or `bazel` is missing, the environment's setup script has not
  run `tools/cloud_setup.sh`.
