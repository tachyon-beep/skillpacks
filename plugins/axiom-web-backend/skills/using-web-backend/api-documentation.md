
# API Documentation

## Overview

**API documentation specialist covering OpenAPI specs, documentation-as-code, testing docs, SDK generation, and preventing documentation debt.**

**Core principle**: Documentation is a product feature that directly impacts developer adoption - invest in keeping it accurate, tested, and discoverable.

## When to Use This Skill

Use when encountering:

- **OpenAPI/Swagger**: Auto-generating docs, customizing Swagger UI, maintaining specs
- **Documentation testing**: Ensuring examples work, preventing stale docs
- **Versioning**: Managing multi-version docs, deprecation notices
- **Documentation-as-code**: Keeping docs in sync with code changes
- **SDK generation**: Generating client libraries from OpenAPI specs
- **Documentation debt**: Detecting and preventing outdated documentation
- **Metrics**: Tracking documentation usage and effectiveness
- **Community docs**: Managing contributions, improving discoverability

**Do NOT use for**:
- General technical writing (see `muna-technical-writer` skill)
- API design principles (see `rest-api-design`, `graphql-api-design`)
- Authentication implementation (see `api-authentication`)

## OpenAPI Specification Best Practices

### Production-Quality OpenAPI Specs

**Complete FastAPI example**:

```python
from fastapi import FastAPI, Path, Query, Body
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI(
    title="Payment Processing API",
    description="""
    # Payment API

    Process payments with PCI-DSS compliance.

    ## Features
    - Multiple payment methods (cards, ACH, digital wallets)
    - Fraud detection
    - Webhook notifications
    - Test mode for development

    ## Rate Limits
    - Standard: 100 requests/minute
    - Premium: 1000 requests/minute

    ## Support
    - Documentation: https://docs.example.com
    - Status: https://status.example.com
    - Support: api-support@example.com
    """,
    version="2.1.0",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "api-support@example.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    },
    servers=[
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "https://sandbox-api.example.com", "description": "Sandbox"}
    ]
)

# Tag organization
tags_metadata = [
    {
        "name": "payments",
        "description": "Payment operations",
        "externalDocs": {
            "description": "Payment Guide",
            "url": "https://docs.example.com/guides/payments"
        }
    }
]

app = FastAPI(openapi_tags=tags_metadata)

# Rich schema with examples
class PaymentRequest(BaseModel):
    amount: float = Field(
        ...,
        gt=0,
        le=999999.99,
        description="Payment amount in USD",
        example=99.99
    )
    currency: str = Field(
        default="USD",
        pattern="^[A-Z]{3}$",
        description="ISO 4217 currency code",
        example="USD"
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "amount": 149.99,
                    "currency": "USD",
                    "payment_method": "card_visa_4242",
                    "description": "Premium subscription"
                },
                {
                    "amount": 29.99,
                    "currency": "EUR",
                    "payment_method": "paypal_account",
                    "description": "Monthly plan"
                }
            ]
        }

# Comprehensive error documentation
@app.post(
    "/payments",
    summary="Create payment",
    description="""
    Creates a new payment transaction.

    ## Processing Time
    Typically 2-5 seconds for card payments.

    ## Idempotency
    Use `Idempotency-Key` header to prevent duplicates.

    ## Test Mode
    Use test payment methods in sandbox environment.
    """,
    responses={
        201: {"description": "Payment created", "model": PaymentResponse},
        400: {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_amount": {
                            "summary": "Amount validation failed",
                            "value": {
                                "error_code": "INVALID_AMOUNT",
                                "message": "Amount must be between 0.01 and 999999.99"
                            }
                        }
                    }
                }
            }
        },
        402: {"description": "Payment declined"},
        429: {"description": "Rate limit exceeded"}
    },
    tags=["payments"]
)
async def create_payment(payment: PaymentRequest):
    pass
```

### Custom OpenAPI Generation

**Add security schemes, custom extensions**:

```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Get your API key at https://dashboard.example.com/api-keys"
        },
        "OAuth2": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": "https://auth.example.com/oauth/authorize",
                    "tokenUrl": "https://auth.example.com/oauth/token",
                    "scopes": {
                        "payments:read": "Read payment data",
                        "payments:write": "Create payments"
                    }
                },
                "clientCredentials": {
                    "tokenUrl": "https://auth.example.com/oauth/token",
                    "scopes": {
                        "payments:read": "Read payment data",
                        "payments:write": "Create payments"
                    }
                }
            }
        }
    }

    # Global security requirement
    openapi_schema["security"] = [{"ApiKeyAuth": []}]

    # Custom extensions for tooling
    openapi_schema["x-api-id"] = "payments-api-v2"
    openapi_schema["x-audience"] = "external"
    openapi_schema["x-ratelimit-default"] = 100

    # Add code samples extension (for Swagger UI)
    for path_data in openapi_schema["paths"].values():
        for operation in path_data.values():
            if isinstance(operation, dict) and "operationId" in operation:
                operation["x-code-samples"] = [
                    {
                        "lang": "curl",
                        "source": generate_curl_example(operation)
                    },
                    {
                        "lang": "python",
                        "source": generate_python_example(operation)
                    }
                ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

## Documentation-as-Code

### Keep Docs in Sync with Code

**Anti-pattern**: Docs in separate repo, manually updated, always stale

**Pattern**: Co-locate docs with code, auto-generate from source

**Implementation**:

```python
# Source of truth: Pydantic models
class PaymentRequest(BaseModel):
    """
    Payment request model.

    Examples:
        Basic payment:
        ```python
        payment = PaymentRequest(
            amount=99.99,
            currency="USD",
            payment_method="pm_card_visa"
        )
        ```
    """
    amount: float = Field(..., description="Amount in USD")
    currency: str = Field(default="USD", description="ISO 4217 currency code")

    class Config:
        schema_extra = {
            "examples": [
                {"amount": 99.99, "currency": "USD", "payment_method": "pm_card_visa"}
            ]
        }

# Docs auto-generated from model
# - OpenAPI spec from Field descriptions
# - Examples from schema_extra
# - Code samples from docstring examples
```

**Prevent schema drift**:

```python
import pytest
from fastapi.testclient import TestClient

def test_openapi_schema_matches_committed():
    """Ensure OpenAPI spec is committed and up-to-date"""
    client = TestClient(app)

    # Get current OpenAPI spec
    current_spec = client.get("/openapi.json").json()

    # Load committed spec
    with open("docs/openapi.json") as f:
        committed_spec = json.load(f)

    # Fail if specs don't match
    assert current_spec == committed_spec, \
        "OpenAPI spec has changed. Run 'make update-openapi-spec' and commit"

def test_all_endpoints_have_examples():
    """Ensure all endpoints have request/response examples"""
    client = TestClient(app)
    spec = client.get("/openapi.json").json()

    for path, methods in spec["paths"].items():
        for method, details in methods.items():
            if method in ["get", "post", "put", "patch", "delete"]:
                # Check request body has example
                if "requestBody" in details:
                    assert "examples" in details["requestBody"]["content"]["application/json"], \
                        f"{method.upper()} {path} missing request examples"

                # Check responses have examples
                for status_code, response in details.get("responses", {}).items():
                    if "content" in response and "application/json" in response["content"]:
                        assert "examples" in response["content"]["application/json"] or \
                               "example" in response["content"]["application/json"]["schema"], \
                               f"{method.upper()} {path} response {status_code} missing examples"
```

### Documentation Pre-Commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Regenerate OpenAPI spec
python -c "
from app.main import app
import json

with open('docs/openapi.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"

# Check if spec changed
git add docs/openapi.json

# Validate spec
npm run validate:openapi

# Run doc tests
pytest tests/test_documentation.py
```

## Documentation Testing

### Ensure Examples Actually Work

**Problem**: Examples in docs become stale, don't work

**Solution**: Test every code example automatically

```python
# Extract examples from OpenAPI spec
import pytest
import requests
from app.main import app

def get_all_examples_from_openapi():
    """Extract all examples from OpenAPI spec"""
    spec = app.openapi()
    examples = []

    for path, methods in spec["paths"].items():
        for method, details in methods.items():
            if "examples" in details.get("requestBody", {}).get("content", {}).get("application/json", {}):
                for example_name, example_data in details["requestBody"]["content"]["application/json"]["examples"].items():
                    examples.append({
                        "path": path,
                        "method": method,
                        "example_name": example_name,
                        "data": example_data["value"]
                    })

    return examples

@pytest.mark.parametrize("example", get_all_examples_from_openapi(), ids=lambda e: f"{e['method']}_{e['path']}_{e['example_name']}")
def test_openapi_examples_are_valid(example, client):
    """Test that all OpenAPI examples are valid requests"""
    method = example["method"]
    path = example["path"]
    data = example["data"]

    response = client.request(method, path, json=data)

    # Examples should either succeed or fail with expected error
    assert response.status_code in [200, 201, 400, 401, 402, 403, 404], \
        f"Example {example['example_name']} for {method.upper()} {path} returned unexpected status {response.status_code}"
```

**Test markdown code samples**:

```python
import pytest
import re
import tempfile
import subprocess

def extract_code_blocks_from_markdown(markdown_file):
    """Extract code blocks from markdown"""
    with open(markdown_file) as f:
        content = f.read()

    # Find code blocks with language
    pattern = r'```(\w+)\n(.*?)```'
    return re.findall(pattern, content, re.DOTALL)

def test_python_examples_in_quickstart():
    """Test that Python examples in quickstart.md execute without errors"""
    code_blocks = extract_code_blocks_from_markdown("docs/quickstart.md")

    for lang, code in code_blocks:
        if lang == "python":
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Replace placeholders
                code = code.replace("sk_test_abc123...", "test_api_key")
                code = code.replace("https://api.example.com", "http://localhost:8000")
                f.write(code)
                f.flush()

                # Run code
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                assert result.returncode == 0, \
                    f"Python example failed:\n{code}\n\nError:\n{result.stderr}"
```

### Documentation Coverage Metrics

```python
def test_documentation_coverage():
    """Ensure all endpoints are documented"""
    from fastapi.openapi.utils import get_openapi

    spec = get_openapi(title="Test", version="1.0.0", routes=app.routes)

    missing_docs = []

    for path, methods in spec["paths"].items():
        for method, details in methods.items():
            # Check summary
            if not details.get("summary"):
                missing_docs.append(f"{method.upper()} {path}: Missing summary")

            # Check description
            if not details.get("description"):
                missing_docs.append(f"{method.upper()} {path}: Missing description")

            # Check examples
            if "requestBody" in details:
                content = details["requestBody"].get("content", {}).get("application/json", {})
                if "examples" not in content and "example" not in content.get("schema", {}):
                    missing_docs.append(f"{method.upper()} {path}: Missing request example")

    assert not missing_docs, \
        f"Documentation incomplete:\n" + "\n".join(missing_docs)
```

## Interactive Documentation

### Swagger UI Customization

**Custom Swagger UI with branding**:

```python
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

app = FastAPI(docs_url=None)  # Disable default docs
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "filter": True,
            "showExtensions": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "defaultModelsExpandDepth": 1,
            "defaultModelExpandDepth": 1
        }
    )
```

**Add "Try It Out" authentication**:

```python
from fastapi.openapi.docs import get_swagger_ui_html

@app.get("/docs")
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="API Docs",
        init_oauth={
            "clientId": "swagger-ui-client",
            "appName": "API Documentation",
            "usePkceWithAuthorizationCodeGrant": True
        }
    )
```

### ReDoc Customization

```python
from fastapi.openapi.docs import get_redoc_html

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="API Documentation - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
        redoc_favicon_url="/static/favicon.png",
        with_google_fonts=True
    )
```

**ReDoc configuration options**:

```html
<!-- static/redoc-config.html -->
<redoc
  spec-url="/openapi.json"
  expand-responses="200,201"
  required-props-first="true"
  sort-props-alphabetically="true"
  hide-download-button="false"
  native-scrollbars="false"
  path-in-middle-panel="true"
  theme='{
    "colors": {
      "primary": {"main": "#32329f"}
    },
    "typography": {
      "fontSize": "14px",
      "fontFamily": "Roboto, sans-serif"
    }
  }'
></redoc>
```

## SDK Generation

### Generate Client SDKs from OpenAPI

**OpenAPI Generator**:

```bash
# Install openapi-generator
npm install -g @openapitools/openapi-generator-cli

# Generate Python SDK
openapi-generator-cli generate \
  -i docs/openapi.json \
  -g python \
  -o sdks/python \
  --additional-properties=packageName=payment_api,projectName=payment-api-python

# Generate TypeScript SDK
openapi-generator-cli generate \
  -i docs/openapi.json \
  -g typescript-fetch \
  -o sdks/typescript \
  --additional-properties=npmName=@example/payment-api,supportsES6=true

# Generate Go SDK
openapi-generator-cli generate \
  -i docs/openapi.json \
  -g go \
  -o sdks/go \
  --additional-properties=packageName=paymentapi
```

**Automate SDK generation in CI**:

```yaml
# .github/workflows/generate-sdks.yml
name: Generate SDKs

on:
  push:
    branches: [main]
    paths:
      - 'docs/openapi.json'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Generate Python SDK
        run: |
          docker run --rm \
            -v ${PWD}:/local \
            openapitools/openapi-generator-cli generate \
            -i /local/docs/openapi.json \
            -g python \
            -o /local/sdks/python

      - name: Test Python SDK
        run: |
          cd sdks/python
          pip install -e .
          pytest

      - name: Publish to PyPI
        if: github.ref == 'refs/heads/main'
        run: |
          cd sdks/python
          python -m build
          twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

**Custom SDK templates**:

```
templates/
├── python/
│   ├── api.mustache           # Custom API client template
│   ├── model.mustache          # Custom model template
│   └── README.mustache         # Custom README
```

```bash
# Generate with custom templates
openapi-generator-cli generate \
  -i docs/openapi.json \
  -g python \
  -o sdks/python \
  -t templates/python \
  --additional-properties=packageName=payment_api
```

## Documentation Versioning

### Version Documentation Separately from API

**Documentation versions**:

```
docs/
├── v1/
│   ├── quickstart.md
│   ├── api-reference.md
│   └── migration-to-v2.md  ← Deprecation notice
├── v2/
│   ├── quickstart.md
│   ├── api-reference.md
│   └── whats-new.md
└── latest -> v2/  # Symlink to current version
```

**Documentation routing**:

```python
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("docs"))

@app.get("/docs")
async def docs_redirect():
    """Redirect to latest docs"""
    return RedirectResponse(url="/docs/v2/")

@app.get("/docs/{version}/{page}")
async def serve_docs(version: str, page: str):
    """Serve versioned documentation"""
    if version not in ["v1", "v2"]:
        raise HTTPException(404)

    # Add deprecation warning for v1
    deprecated = version == "v1"

    template = env.get_template(f"{version}/{page}.md")
    content = template.render(deprecated=deprecated)

    return HTMLResponse(content)
```

**Deprecation banner**:

```html
<!-- docs/templates/base.html -->
{% if deprecated %}
<div class="deprecation-banner">
  ⚠️ <strong>Deprecated</strong>: This documentation is for API v1,
  which will be sunset on June 1, 2025.
  <a href="/docs/v2/migration">Migrate to v2</a>
</div>
{% endif %}
```

## Documentation Debt Detection

### Prevent Stale Documentation

**Detect outdated docs**:

```python
import pytest
from datetime import datetime, timedelta

def test_documentation_freshness():
    """Ensure docs have been updated recently"""
    docs_modified = datetime.fromtimestamp(
        os.path.getmtime("docs/api-reference.md")
    )

    # Fail if docs haven't been updated in 90 days
    max_age = timedelta(days=90)
    age = datetime.now() - docs_modified

    assert age < max_age, \
        f"API docs are {age.days} days old. Review and update or add exemption comment."
```

**Track documentation TODOs**:

```python
def test_no_documentation_todos():
    """Ensure no TODO comments in docs"""
    import re

    doc_files = glob.glob("docs/**/*.md", recursive=True)
    todos = []

    for doc_file in doc_files:
        with open(doc_file) as f:
            for line_num, line in enumerate(f, 1):
                if re.search(r'TODO|FIXME|XXX', line):
                    todos.append(f"{doc_file}:{line_num}: {line.strip()}")

    assert not todos, \
        f"Documentation has {len(todos)} TODOs:\n" + "\n".join(todos)
```

**Broken link detection**:

```python
import pytest
import requests
from bs4 import BeautifulSoup
import re

def extract_links_from_markdown(markdown_file):
    """Extract all HTTP(S) links from markdown"""
    with open(markdown_file) as f:
        content = f.read()

    # Find markdown links [text](url)
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    return [(text, url) for text, url in links if url.startswith('http')]

def test_no_broken_links_in_docs():
    """Ensure all external links in docs are valid"""
    doc_files = glob.glob("docs/**/*.md", recursive=True)
    broken_links = []

    for doc_file in doc_files:
        for text, url in extract_links_from_markdown(doc_file):
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                if response.status_code >= 400:
                    broken_links.append(f"{doc_file}: {url} ({response.status_code})")
            except requests.RequestException as e:
                broken_links.append(f"{doc_file}: {url} (error: {e})")

    assert not broken_links, \
        f"Found {len(broken_links)} broken links:\n" + "\n".join(broken_links)
```

## Documentation Metrics

### Track Documentation Usage

**Analytics integration**:

```python
from fastapi import Request
import analytics

@app.middleware("http")
async def track_doc_views(request: Request, call_next):
    if request.url.path.startswith("/docs"):
        # Track page view
        analytics.track(
            user_id="anonymous",
            event="Documentation Viewed",
            properties={
                "page": request.url.path,
                "version": request.url.path.split("/")[2] if len(request.url.path.split("/")) > 2 else "latest",
                "referrer": request.headers.get("referer")
            }
        )

    return await call_next(request)
```

**Track "Try It Out" usage**:

```javascript
// Inject into Swagger UI
const originalExecute = swagger.presets.apis.execute;
swagger.presets.apis.execute = function(spec) {
  // Track API call from docs
  analytics.track('API Call from Docs', {
    endpoint: spec.path,
    method: spec.method,
    success: spec.response.status < 400
  });

  return originalExecute(spec);
};
```

**Documentation health dashboard**:

```python
from fastapi import APIRouter
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/admin/docs-metrics")
async def get_doc_metrics(db: Session = Depends(get_db)):
    """Dashboard for documentation health"""

    # Page views by version
    views_by_version = analytics.query(
        "Documentation Viewed",
        group_by="version",
        since=datetime.now() - timedelta(days=30)
    )

    # Most viewed pages
    top_pages = analytics.query(
        "Documentation Viewed",
        group_by="page",
        since=datetime.now() - timedelta(days=30),
        limit=10
    )

    # Try it out usage
    api_calls = analytics.query(
        "API Call from Docs",
        since=datetime.now() - timedelta(days=30)
    )

    # Documentation freshness
    freshness = {
        "quickstart.md": get_file_age("docs/quickstart.md"),
        "api-reference.md": get_file_age("docs/api-reference.md")
    }

    return {
        "views_by_version": views_by_version,
        "top_pages": top_pages,
        "api_calls_from_docs": api_calls,
        "freshness": freshness,
        "health_score": calculate_doc_health_score()
    }

def calculate_doc_health_score():
    """Calculate documentation health (0-100)"""
    score = 100

    # Deduct for stale docs (>90 days old)
    for doc_file in glob.glob("docs/**/*.md", recursive=True):
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(doc_file))).days
        if age_days > 90:
            score -= 10

    # Deduct for broken links
    broken_links = count_broken_links()
    score -= min(broken_links * 5, 30)

    # Deduct for missing examples
    endpoints_without_examples = count_endpoints_without_examples()
    score -= min(endpoints_without_examples * 3, 20)

    return max(score, 0)
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **Docs in separate repo** | Always out of sync | Co-locate with code |
| **Manual example updates** | Examples become stale | Test examples in CI |
| **No deprecation notices** | Breaking changes surprise users | Document deprecation 6+ months ahead |
| **Generic descriptions** | Doesn't help developers | Specific use cases, edge cases |
| **No versioned docs** | Can't reference old versions | Version docs separately |
| **Untested SDKs** | Generated SDKs don't work | Test generated SDKs in CI |
| **No documentation metrics** | Can't measure effectiveness | Track page views, usage |
| **Single example per endpoint** | Doesn't show edge cases | Multiple examples (success, errors) |

## Cross-References

**Related skills**:
- **Technical writing** → `muna-technical-writer` (writing style, organization)
- **API design** → `rest-api-design`, `graphql-api-design` (design patterns)
- **API testing** → `api-testing` (contract testing, examples)
- **Authentication** → `api-authentication` (auth flow documentation)

## Further Reading

- **OpenAPI Specification**: https://spec.openapis.org/oas/v3.1.0
- **FastAPI docs**: https://fastapi.tiangolo.com/tutorial/metadata/
- **Swagger UI**: https://swagger.io/docs/open-source-tools/swagger-ui/
- **ReDoc**: https://redoc.ly/docs/
- **Write the Docs**: https://www.writethedocs.org/
