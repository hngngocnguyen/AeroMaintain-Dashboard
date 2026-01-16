#!/usr/bin/env python3
"""
AeroMaintain Streamlit Dashboard - Pre-flight Verification Script
Checks all dependencies and files before launching the app
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Verify Python version is 3.10 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Current: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_packages():
    """Check if required packages are installed"""
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sklearn',
        'xgboost',
        'scipy'
    ]
    
    missing = []
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("→ Install with: pip install -r streamlit_requirements.txt\n")
        return False
    return True

def check_files():
    """Check if required files exist"""
    project_path = Path(__file__).parent
    
    files_to_check = {
        'app.py': 'Main Streamlit application',
        'streamlit_requirements.txt': 'Python dependencies',
        '.streamlit/config.toml': 'Streamlit configuration',
    }
    
    all_exist = True
    for filename, description in files_to_check.items():
        filepath = project_path / filename
        if filepath.exists():
            print(f"✅ {filename:35} ({description})")
        else:
            print(f"❌ {filename:35} - MISSING")
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check if dataset files exist"""
    project_path = Path(__file__).parent
    dataset_path = project_path / 'dataset'
    
    required_files = {
        'train_FD001.txt': 'Training data',
        'test_FD001.txt': 'Test data',
        'RUL_FD001.txt': 'RUL values'
    }
    
    all_exist = True
    
    if not dataset_path.exists():
        print("❌ dataset/ folder not found")
        return False
    
    print("\nDataset files:")
    for filename, description in required_files.items():
        filepath = dataset_path / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"✅ {filename:20} ({size:.2f} MB) - {description}")
        else:
            print(f"❌ {filename:20} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks"""
    print("\n" + "="*70)
    print("🛩️  AeroMaintain Streamlit Dashboard - Pre-flight Verification")
    print("="*70 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_packages),
        ("Project Files", check_files),
        ("Dataset Files", check_dataset),
    ]
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 70)
        results[check_name] = check_func()
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY:")
    print("="*70)
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {check_name}")
    
    print("="*70)
    
    if all_passed:
        print("\n✅ All checks passed!")
        print("\n🚀 Ready to launch the dashboard:")
        print("   streamlit run app.py")
        print("\n📊 Access at: http://localhost:8501\n")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n📖 For help, see: GETTING_STARTED.md\n")
        return 1

if __name__ == '__main__':
    exit(main())
