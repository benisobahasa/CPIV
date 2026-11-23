# Repository setup checklist (GitHub)

Files in this repository configure CI, CodeQL, releases, Dependabot, issue/PR templates and
the security policy. A few things can only be switched on in the GitHub web interface.

## After the first push
1. **Actions tab**: confirm the `CI` and `CodeQL` workflows run. If a job fails, open it and
   read the log.
2. **Settings > General**: add a description, website and topics (for example `speech`,
   `forensics`, `explainable-ai`, `benchmark`, `voice`); tick **Discussions** if you want
   questions outside issues.
3. **Settings > Code security**:
   - enable **Private vulnerability reporting** (used by `SECURITY.md`)
   - enable **Dependabot alerts** and **Dependabot security updates**
   - enable **Secret scanning** and **Push protection**
4. **Settings > Branches**: add a branch protection rule for `main`: require a pull request,
   require the status checks `test` and `build and check package`, block force pushes.

## Making a release
1. Update `version` in `pyproject.toml`, `src/cpiv/__init__.py` and `CITATION.cff`, and add
   an entry to `CHANGELOG.md`.
2. Commit and push, then tag:
   ```bash
   git tag v0.2.1
   git push origin v0.2.1
   ```
3. The `Release` workflow checks that the tag matches the package version, builds the wheel
   and source archive, and publishes a GitHub Release with generated notes.

## Optional: publish to PyPI
1. Create an account at pypi.org and add a **pending trusted publisher** for the project
   `cpiv` with: owner `benisobahasa`, repository `CPIV`, workflow `release.yml`,
   environment `pypi`. Check that the name `cpiv` is available on PyPI.
2. In GitHub: **Settings > Environments** create an environment named `pypi`.
3. In GitHub: **Settings > Secrets and variables > Actions > Variables** add
   `PUBLISH_PYPI` with value `true`.
4. The next tagged release also uploads to PyPI, and users can run `pip install cpiv`.

No API tokens are stored: PyPI trusted publishing uses short-lived OIDC credentials.
