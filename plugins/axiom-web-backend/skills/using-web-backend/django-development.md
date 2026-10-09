# Django Version-aware Recipe

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Read project settings, supported Django/DRF versions, middleware/auth and database routing. Use current project conventions for models/views/serializers.

Check transaction scope, query/index shape, permission checks (including object level), validation and response/error compatibility. Inspect sync/async boundary and connection lifecycle before adopting async patterns.

Test migrations and affected request/admin/background paths where relevant. Use [Django docs](https://docs.djangoproject.com/) for the supported version and [DRF docs](https://www.django-rest-framework.org/) when applicable. Do not add settings/dependencies from an old tutorial without a requirement.
