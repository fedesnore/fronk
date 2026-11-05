# Backend modules

Each subdirectory contains one `backend.json`, its source and its explanation.
Run `python3 tools/backend-module.py new mycpu` from the repository root to
create a working scalar example. See [BACKEND_MODULES.md](../../docs/BACKEND_MODULES.md)
for building, testing, installer integration and submitting a module.

Modules are compiled into the ordinary `fr_backend` table. There is no dynamic
loader or per-request module lookup. A module owns expert computation only;
networking, extraction and head scheduling keep their existing boundaries.
