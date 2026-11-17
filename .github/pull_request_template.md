## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] Every commit is signed off (`git commit -s`)
- [ ] No secrets, keys, personal data or internal addresses in the diff
- [ ] If I touched `deploy/`, `config/`, `.env.example` or the `Dockerfile`: the end-to-end suite passes (`cd e2e && pytest`) and the matching docs page is still true
- [ ] If I changed `synap/`: it fixes an issue labelled `approved` (see [CONTRIBUTING.md](../CONTRIBUTING.md))
