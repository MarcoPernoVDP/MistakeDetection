# Setup

## Virtual Environment

Create a virtual environment:
```bash
virtualenv venv --python=3.12
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Useful Commands

Update pip:
```bash
python -m pip install --upgrade pip
```

Freeze dependencies:
```bash
pip freeze > requirements.txt
```

Install a specific package:
```bash
pip install package_name
```

Uninstall a package:
```bash
pip uninstall package_name
```

List installed packages:
```bash
pip list
```