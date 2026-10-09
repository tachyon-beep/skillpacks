# FastAPI Version-aware Recipe

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Inspect `pyproject.toml`/lockfile and actual ASGI lifecycle/dependencies before changing routes. Follow the installed FastAPI/Pydantic versions, not a copied scaffold.

Check dependency/auth scope, request/response validation, exception mapping and generated schema against real behavior. Avoid blocking I/O/CPU work on the event loop; choose supported thread/process/task integration with explicit cancellation/resource ownership.

Test lifespan/startup/shutdown and database/session cleanup, including errors. Use [FastAPI docs](https://fastapi.tiangolo.com/) and installed-version primary release notes for changed APIs. Preserve existing application layout unless the task requires restructuring.
