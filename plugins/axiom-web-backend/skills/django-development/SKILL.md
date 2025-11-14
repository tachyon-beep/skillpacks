---
name: django-development
description: Use when building Django REST APIs, optimizing Django ORM queries, structuring DRF serializers, managing migrations, implementing caching, or deploying Django apps - covers production patterns for Django + DRF beyond basic tutorials
---

# Django Development

## Overview

**Django development specialist covering Django ORM optimization, DRF best practices, caching strategies, migrations, testing, and production deployment.**

**Core principle**: Django's "batteries included" philosophy is powerful but requires understanding which battery to use when - master Django's tools to avoid reinventing wheels or choosing wrong patterns.

## When to Use This Skill

Use when encountering:

- **ORM optimization**: N+1 queries, select_related vs prefetch_related, query performance
- **DRF patterns**: Serializers, ViewSets, permissions, nested relationships
- **Caching**: Cache framework, per-view caching, template fragment caching
- **Migrations**: Zero-downtime migrations, data migrations, squashing
- **Testing**: Django TestCase, fixtures, factories, mocking
- **Deployment**: Gunicorn, static files, database pooling
- **Async Django**: Channels, async views, WebSockets
- **Admin customization**: Custom admin actions, list filters, inlines

**Do NOT use for**:
- General Python patterns (use `axiom-python-engineering`)
- API design principles (use `rest-api-design`)
- Database-agnostic patterns (use `database-integration`)
- Authentication flows (use `api-authentication`)

## Django ORM Optimization

### select_related vs prefetch_related

**Decision matrix**:

| Relationship | Method | SQL Strategy | Use When |
|--------------|--------|--------------|----------|
| ForeignKey (many-to-one) | `select_related` | JOIN | Book → Author |
| OneToOneField | `select_related` | JOIN | User → Profile |
| Reverse ForeignKey (one-to-many) | `prefetch_related` | Separate query + IN | Author → Books |
| ManyToManyField | `prefetch_related` | Separate query + IN | Book → Tags |

**Example - select_related (JOIN)**:

```python
# BAD: N+1 queries (1 + N)
books = Book.objects.all()
for book in books:
    print(book.author.name)  # Additional query per book

# GOOD: Single JOIN query
books = Book.objects.select_related('author').all()
for book in books:
    print(book.author.name)  # No additional queries

# SQL generated:
# SELECT book.*, author.* FROM book JOIN author ON book.author_id = author.id
```

**Example - prefetch_related (IN query)**:

```python
# BAD: N+1 queries
authors = Author.objects.all()
for author in authors:
    print(author.books.count())  # Query per author

# GOOD: 2 queries total
authors = Author.objects.prefetch_related('books').all()
for author in authors:
    print(author.books.count())  # No additional queries

# SQL generated:
# Query 1: SELECT * FROM author
# Query 2: SELECT * FROM book WHERE author_id IN (1, 2, 3, ...)
```

**Nested prefetching**:

```python
from django.db.models import Prefetch

# Fetch authors → books → reviews (3 queries)
authors = Author.objects.prefetch_related(
    Prefetch('books', queryset=Book.objects.prefetch_related('reviews'))
)

# Custom filtering on prefetch
recent_books = Book.objects.filter(
    published_date__gte=timezone.now() - timedelta(days=30)
).order_by('-published_date')

authors = Author.objects.prefetch_related(
    Prefetch('books', queryset=recent_books, to_attr='recent_books')
)

# Access via custom attribute
for author in authors:
    for book in author.recent_books:  # Only recent books
        print(book.title)
```

### Query Debugging

```python
from django.db import connection, reset_queries
from django.conf import settings

# Enable in settings.py: DEBUG = True
# Or use django-debug-toolbar

def debug_queries(func):
    """Decorator to debug query counts"""
    def wrapper(*args, **kwargs):
        reset_queries()
        result = func(*args, **kwargs)
        print(f"Queries: {len(connection.queries)}")
        for query in connection.queries:
            print(f"  {query['time']}s: {query['sql'][:100]}")
        return result
    return wrapper

@debug_queries
def get_books():
    return list(Book.objects.select_related('author').prefetch_related('tags'))
```

**Django Debug Toolbar** (production alternative - django-silk):

```python
# settings.py
INSTALLED_APPS = [
    'debug_toolbar',
    # ...
]

MIDDLEWARE = [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    # ...
]

INTERNAL_IPS = ['127.0.0.1']

# For production: use django-silk for profiling
INSTALLED_APPS += ['silk']
MIDDLEWARE += ['silk.middleware.SilkyMiddleware']
```

### Annotation and Aggregation

**Annotate** (add computed fields):

```python
from django.db.models import Count, Avg, Sum, F, Q

# Add book count to each author
authors = Author.objects.annotate(
    book_count=Count('books'),
    avg_rating=Avg('books__rating'),
    total_sales=Sum('books__sales')
)

for author in authors:
    print(f"{author.name}: {author.book_count} books, avg rating {author.avg_rating}")
```

**Aggregate** (single value across queryset):

```python
from django.db.models import Avg

# Get average rating across all books
avg_rating = Book.objects.aggregate(Avg('rating'))
# Returns: {'rating__avg': 4.2}

# Multiple aggregations
stats = Book.objects.aggregate(
    avg_rating=Avg('rating'),
    total_sales=Sum('sales'),
    book_count=Count('id')
)
```

**Conditional aggregation with Q**:

```python
from django.db.models import Q, Count

# Count books by rating category
Author.objects.annotate(
    high_rated_books=Count('books', filter=Q(books__rating__gte=4.0)),
    low_rated_books=Count('books', filter=Q(books__rating__lt=3.0))
)
```

## Django REST Framework Patterns

### ViewSet vs APIView

**Decision matrix**:

| Use | Pattern | When |
|-----|---------|------|
| Standard CRUD | `ModelViewSet` | Full REST API for model |
| Custom actions only | `ViewSet` | Non-standard endpoints |
| Read-only API | `ReadOnlyModelViewSet` | GET/LIST only |
| Fine control | `APIView` or `@api_view` | Custom business logic |

**ModelViewSet** (full CRUD):

```python
from rest_framework import viewsets, filters
from rest_framework.decorators import action
from rest_framework.response import Response

class BookViewSet(viewsets.ModelViewSet):
    """
    Provides: list, create, retrieve, update, partial_update, destroy
    """
    queryset = Book.objects.select_related('author').prefetch_related('tags')
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['title', 'author__name']
    ordering_fields = ['published_date', 'rating']

    def get_queryset(self):
        """Optimize queryset based on action"""
        queryset = super().get_queryset()

        if self.action == 'list':
            # List doesn't need full detail
            return queryset.only('id', 'title', 'author__name')

        return queryset

    @action(detail=True, methods=['post'])
    def publish(self, request, pk=None):
        """Custom action: POST /books/123/publish/"""
        book = self.get_object()
        book.status = 'published'
        book.published_date = timezone.now()
        book.save()
        return Response({'status': 'published'})

    @action(detail=False, methods=['get'])
    def bestsellers(self, request):
        """Custom list action: GET /books/bestsellers/"""
        books = self.get_queryset().filter(sales__gte=10000).order_by('-sales')[:10]
        serializer = self.get_serializer(books, many=True)
        return Response(serializer.data)
```

### Serializer Patterns

**Basic serializer with validation**:

```python
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password]
    )
    password_confirm = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password', 'password_confirm']
        read_only_fields = ['id']

    # Field-level validation
    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("Email already in use")
        return value.lower()

    # Object-level validation (cross-field)
    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                'password_confirm': "Passwords don't match"
            })
        attrs.pop('password_confirm')
        return attrs

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create(**validated_data)
        user.set_password(password)
        user.save()
        return user
```

**Nested serializers (read-only)**:

```python
class AuthorSerializer(serializers.ModelSerializer):
    book_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Author
        fields = ['id', 'name', 'bio', 'book_count']

class BookSerializer(serializers.ModelSerializer):
    author = AuthorSerializer(read_only=True)
    author_id = serializers.PrimaryKeyRelatedField(
        queryset=Author.objects.all(),
        source='author',
        write_only=True
    )

    class Meta:
        model = Book
        fields = ['id', 'title', 'author', 'author_id', 'published_date']
```

**Dynamic fields** (include/exclude fields via query params):

```python
class DynamicFieldsModelSerializer(serializers.ModelSerializer):
    """
    Usage: /api/books/?fields=id,title,author
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        request = self.context.get('request')
        if request:
            fields = request.query_params.get('fields')
            if fields:
                fields = fields.split(',')
                allowed = set(fields)
                existing = set(self.fields.keys())
                for field_name in existing - allowed:
                    self.fields.pop(field_name)

class BookSerializer(DynamicFieldsModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
```

## Django Caching

### Cache Framework Setup

```python
# settings.py

# Redis cache (production)
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {'max_connections': 50},
            'PARSER_CLASS': 'redis.connection.HiredisParser',
        },
        'KEY_PREFIX': 'myapp',
        'TIMEOUT': 300,  # Default 5 minutes
    }
}

# Memcached (alternative)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.PyMemcacheCache',
        'LOCATION': '127.0.0.1:11211',
    }
}

# Local memory (development only)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}
```

### Per-View Caching

```python
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator

# Function-based view
@cache_page(60 * 15)  # Cache for 15 minutes
def book_list(request):
    books = Book.objects.all()
    return render(request, 'books/list.html', {'books': books})

# Class-based view
class BookListView(ListView):
    model = Book

    @method_decorator(cache_page(60 * 15))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

# DRF ViewSet
from rest_framework_extensions.cache.decorators import cache_response

class BookViewSet(viewsets.ModelViewSet):
    @cache_response(timeout=60*15, key_func='calculate_cache_key')
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    def calculate_cache_key(self, view_instance, view_method, request, args, kwargs):
        # Custom cache key including user, filters
        return f"books:list:{request.user.id}:{request.GET.urlencode()}"
```

### Low-Level Cache API

```python
from django.core.cache import cache

# Set cache
cache.set('my_key', 'my_value', timeout=300)

# Get cache
value = cache.get('my_key')
if value is None:
    value = expensive_computation()
    cache.set('my_key', value, timeout=300)

# Get or set (atomic)
value = cache.get_or_set('my_key', lambda: expensive_computation(), timeout=300)

# Delete cache
cache.delete('my_key')

# Clear all
cache.clear()

# Multiple keys
cache.set_many({'key1': 'value1', 'key2': 'value2'}, timeout=300)
values = cache.get_many(['key1', 'key2'])

# Increment/decrement
cache.set('counter', 0)
cache.incr('counter')  # 1
cache.incr('counter', delta=5)  # 6
```

### Cache Invalidation Patterns

```python
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

@receiver([post_save, post_delete], sender=Book)
def invalidate_book_cache(sender, instance, **kwargs):
    """Invalidate cache when book changes"""
    cache.delete(f'book:{instance.id}')
    cache.delete('books:list')  # Invalidate list cache
    cache.delete(f'author:{instance.author_id}:books')

# Pattern: Cache with version tags
def get_books():
    version = cache.get('books:version', 0)
    cache_key = f'books:list:v{version}'
    books = cache.get(cache_key)

    if books is None:
        books = list(Book.objects.all())
        cache.set(cache_key, books, timeout=3600)

    return books

def invalidate_books():
    """Bump version to invalidate all book caches"""
    version = cache.get('books:version', 0)
    cache.set('books:version', version + 1)
```

## Django Migrations

### Zero-Downtime Migration Pattern

**Adding NOT NULL column to large table**:

```python
# Step 1: Add nullable field (migration 0002)
class Migration(migrations.Migration):
    operations = [
        migrations.AddField(
            model_name='user',
            name='department',
            field=models.CharField(max_length=100, null=True, blank=True),
        ),
    ]

# Step 2: Populate data in batches (migration 0003)
from django.db import migrations

def populate_department(apps, schema_editor):
    User = apps.get_model('myapp', 'User')

    # Batch update for performance
    batch_size = 10000
    total = User.objects.filter(department__isnull=True).count()

    for offset in range(0, total, batch_size):
        users = User.objects.filter(department__isnull=True)[offset:offset+batch_size]
        for user in users:
            user.department = determine_department(user)  # Your logic
        User.objects.bulk_update(users, ['department'], batch_size=batch_size)

class Migration(migrations.Migration):
    dependencies = [('myapp', '0002_add_department')],
    operations = [
        migrations.RunPython(populate_department, migrations.RunPython.noop),
    ]

# Step 3: Make NOT NULL (migration 0004)
class Migration(migrations.Migration):
    dependencies = [('myapp', '0003_populate_department')],
    operations = [
        migrations.AlterField(
            model_name='user',
            name='department',
            field=models.CharField(max_length=100),  # NOT NULL
        ),
    ]
```

### Concurrent Index Creation (PostgreSQL)

```python
from django.contrib.postgres.operations import AddIndexConcurrently
from django.db import migrations, models

class Migration(migrations.Migration):
    atomic = False  # Required for CONCURRENTLY operations

    operations = [
        AddIndexConcurrently(
            model_name='book',
            index=models.Index(fields=['published_date'], name='book_published_idx'),
        ),
    ]
```

### Squashing Migrations

```bash
# Squash migrations 0001 through 0020 into single migration
python manage.py squashmigrations myapp 0001 0020

# This creates migrations/0001_squashed_0020.py
# After deploying squashed migration, delete originals:
# migrations/0001.py through migrations/0020.py
```

## Django Testing

### TestCase vs TransactionTestCase

| Feature | TestCase | TransactionTestCase |
|---------|----------|---------------------|
| Speed | Fast (no DB reset between tests) | Slow (resets DB each test) |
| Transactions | Wrapped in transaction, rolled back | No automatic transaction |
| Use for | Most tests | Testing transaction behavior, signals |

**Example - TestCase**:

```python
from django.test import TestCase
from myapp.models import Book

class BookModelTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        """Run once for entire test class (fast)"""
        cls.author = Author.objects.create(name="Test Author")

    def setUp(self):
        """Run before each test method"""
        self.book = Book.objects.create(
            title="Test Book",
            author=self.author
        )

    def test_book_str(self):
        self.assertEqual(str(self.book), "Test Book")

    def test_book_author_relationship(self):
        self.assertEqual(self.book.author.name, "Test Author")
```

### API Testing with DRF

```python
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from django.contrib.auth.models import User

class BookAPITest(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.book = Book.objects.create(title="Test Book")

    def test_list_books_unauthenticated(self):
        response = self.client.get('/api/books/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_create_book_authenticated(self):
        self.client.force_authenticate(user=self.user)
        data = {'title': 'New Book', 'author': self.author.id}
        response = self.client.post('/api/books/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Book.objects.count(), 2)

    def test_update_book_unauthorized(self):
        other_user = User.objects.create_user(username='other', password='pass')
        self.client.force_authenticate(user=other_user)
        data = {'title': 'Updated Title'}
        response = self.client.patch(f'/api/books/{self.book.id}/', data)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
```

### Factory Pattern with factory_boy

```python
# tests/factories.py
import factory
from myapp.models import Author, Book

class AuthorFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Author

    name = factory.Faker('name')
    bio = factory.Faker('text', max_nb_chars=200)

class BookFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Book

    title = factory.Faker('sentence', nb_words=4)
    author = factory.SubFactory(AuthorFactory)
    published_date = factory.Faker('date_this_decade')
    isbn = factory.Sequence(lambda n: f'978-0-{n:09d}')

# Usage in tests
class BookTest(TestCase):
    def test_book_creation(self):
        book = BookFactory.create()  # Creates Author too
        self.assertIsNotNone(book.id)

    def test_multiple_books(self):
        books = BookFactory.create_batch(10)  # Create 10 books
        self.assertEqual(len(books), 10)

    def test_author_with_books(self):
        author = AuthorFactory.create()
        BookFactory.create_batch(5, author=author)
        self.assertEqual(author.books.count(), 5)
```

## Django Settings Organization

### Multiple Environment Configs

```
myproject/
└── settings/
    ├── __init__.py
    ├── base.py          # Common settings
    ├── development.py   # Dev overrides
    ├── production.py    # Prod overrides
    └── test.py          # Test overrides
```

**settings/base.py**:

```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')

INSTALLED_APPS = [
    'django.contrib.admin',
    # ...
    'rest_framework',
    'myapp',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}
```

**settings/development.py**:

```python
from .base import *

DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Use console email backend
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Local cache
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    }
}

# Debug toolbar
INSTALLED_APPS += ['debug_toolbar']
MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
INTERNAL_IPS = ['127.0.0.1']
```

**settings/production.py**:

```python
from .base import *

DEBUG = False

ALLOWED_HOSTS = [os.environ.get('ALLOWED_HOST')]

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Redis cache
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL'),
    }
}

# Real email
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USE_TLS = True
```

**Usage**:

```bash
# Development
export DJANGO_SETTINGS_MODULE=myproject.settings.development
python manage.py runserver

# Production
export DJANGO_SETTINGS_MODULE=myproject.settings.production
gunicorn myproject.wsgi:application
```

## Django Deployment

### Gunicorn Configuration

```python
# gunicorn_config.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"  # or "gevent" for async
worker_connections = 1000
max_requests = 1000  # Restart workers after N requests (prevent memory leaks)
max_requests_jitter = 100
timeout = 30
keepalive = 2

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Process naming
proc_name = "myproject"

# Server mechanics
daemon = False
pidfile = "/var/run/gunicorn.pid"
```

**Systemd service**:

```ini
# /etc/systemd/system/myproject.service
[Unit]
Description=MyProject Django Application
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/myproject
Environment="DJANGO_SETTINGS_MODULE=myproject.settings.production"
ExecStart=/var/www/myproject/venv/bin/gunicorn \
    --config /var/www/myproject/gunicorn_config.py \
    myproject.wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always

[Install]
WantedBy=multi-user.target
```

### Static and Media Files

```python
# settings/production.py
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Use WhiteNoise for static files
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # After SecurityMiddleware
    # ...
]

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
```

**Collect static files**:

```bash
python manage.py collectstatic --noinput
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **Lazy loading in loops** | N+1 queries | Use `select_related`/`prefetch_related` |
| **No database indexing** | Slow queries | Add `db_index=True` or Meta indexes |
| **Signals for async work** | Blocks requests | Use Celery tasks instead |
| **Generic serializers for everything** | Over-fetching data | Create optimized serializers per use case |
| **No caching** | Repeated expensive queries | Cache querysets, views, template fragments |
| **Migrations in production without testing** | Downtime, data loss | Test on production-sized datasets first |
| **DEBUG=True in production** | Security risk, slow | Always DEBUG=False in production |
| **No connection pooling** | Exhausts DB connections | Use pgBouncer or django-db-geventpool |

## Cross-References

**Related skills**:
- **Database optimization** → `database-integration` (connection pooling, migrations)
- **API testing** → `api-testing` (DRF testing patterns)
- **Authentication** → `api-authentication` (DRF token auth, JWT)
- **REST API design** → `rest-api-design` (API patterns)

## Further Reading

- **Django docs**: https://docs.djangoproject.com/
- **DRF docs**: https://www.django-rest-framework.org/
- **Two Scoops of Django**: Best practices book
- **Classy Class-Based Views**: https://ccbv.co.uk/
- **Classy Django REST Framework**: https://www.cdrf.co/
