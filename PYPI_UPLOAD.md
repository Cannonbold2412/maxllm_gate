# Uploading this package to PyPI

This guide shows the safest release flow: build once, test on TestPyPI, then publish to PyPI.

## 1) Prepare your environment

Use Python 3.9+ and install packaging tools:

```bash
python -m pip install --upgrade pip build twine
```

## 2) Bump the package version

Update the version in `pyproject.toml` before each release.

Example:

```toml
[project]
version = "0.1.1"
```

## 3) Clean old artifacts

```bash
python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['dist','build']]; [f.unlink() for f in pathlib.Path('.').glob('*.egg-info')]"
```

## 4) Build distributions

```bash
python -m build
```

This should create files in `dist/`:
- `*.tar.gz` (sdist)
- `*.whl` (wheel)

## 5) Validate package metadata

```bash
python -m twine check dist/*
```

## 6) Configure PyPI credentials

### Option A: API token via environment variable (recommended)

Windows PowerShell:

```powershell
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="pypi-<your-token>"
```

For TestPyPI token, use the TestPyPI token value instead.

### Option B: `~/.pypirc`

Create `%USERPROFILE%\.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-<your-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-testpypi-token>
```

## 7) Upload to TestPyPI first

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI to verify:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple maxllm_gate
```

## 8) Upload to production PyPI

```bash
python -m twine upload dist/*
```

## 9) Verify release

- Open your project page on PyPI.
- Confirm the new version is visible.
- Test install in a clean virtual environment.

## 10) Common errors

- **`File already exists`**: version already published. Bump version and rebuild.
- **Auth errors**: token invalid or missing `__token__` username.
- **`isn't allowed to upload to project`**: the package name is already owned by another PyPI project. Rename `project.name` in `pyproject.toml`.
- **Metadata check fails**: run `twine check` and fix `pyproject.toml` metadata.

## Recommended release checklist

1. Run tests locally.
2. Bump version.
3. Build + `twine check`.
4. Upload to TestPyPI.
5. Smoke test install.
6. Upload to PyPI.
