"""
Dependency Analyzer for ForestGaps-DL

This script analyzes the project to extract imported packages and dependencies.
It generates a comprehensive requirements.txt file based on the analysis.
"""

import os
import ast
import re
import importlib
import pkg_resources
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Regular expressions to find imports
IMPORT_RE = re.compile(r"^import\s+([a-zA-Z0-9_.]+)")
FROM_IMPORT_RE = re.compile(r"^from\s+([a-zA-Z0-9_.]+)\s+import")


class DependencyAnalyzer:
    """Analyzes Python files to extract dependency information."""

    def __init__(self, project_root: str):
        """
        Initialize the dependency analyzer.
        
        Args:
            project_root: Root directory of the project to analyze
        """
        self.project_root = Path(project_root)
        self.all_imports: Set[str] = set()
        self.import_counts: Dict[str, int] = defaultdict(int)
        self.package_name = "forestgaps_dl"  # Default package name
        self.standard_library = self._get_standard_library()
        self.encoding_errors = 0
        
    def _get_standard_library(self) -> Set[str]:
        """Get a set of Python standard library module names."""
        # Alternate implementation that doesn't require stdlib_list
        import sys
        import pkgutil
        
        stdlib_modules = set()
        
        # Add modules from sys.builtin_module_names
        stdlib_modules.update(sys.builtin_module_names)
        
        # Add modules from standard library paths
        for path in sys.path:
            if "site-packages" not in path and "dist-packages" not in path:
                try:
                    for module_info in pkgutil.iter_modules([path]):
                        stdlib_modules.add(module_info.name)
                except (ImportError, OSError):
                    continue
        
        # Add some common standard libraries that might be missed
        common_stdlib = {
            "abc", "argparse", "ast", "asyncio", "base64", "collections", "concurrent", 
            "contextlib", "copy", "csv", "ctypes", "datetime", "decimal", "difflib", 
            "email", "enum", "functools", "gettext", "glob", "gzip", "hashlib", 
            "http", "importlib", "inspect", "io", "itertools", "json", "logging", 
            "math", "multiprocessing", "operator", "os", "pathlib", "pickle", 
            "platform", "pprint", "random", "re", "shutil", "signal", "socket", 
            "sqlite3", "ssl", "statistics", "string", "subprocess", "sys", "tempfile", 
            "threading", "time", "traceback", "typing", "unittest", "urllib", "uuid", 
            "warnings", "weakref", "xml", "zipfile"
        }
        stdlib_modules.update(common_stdlib)
        
        return stdlib_modules
    
    def _extract_package_name(self, import_path: str) -> str:
        """Extract the top-level package name from an import path."""
        return import_path.split('.')[0]
    
    def _is_project_module(self, package_name: str) -> bool:
        """Check if a package is part of the project."""
        return package_name == self.package_name
    
    def _is_standard_library(self, package_name: str) -> bool:
        """Check if a package is part of the Python standard library."""
        return package_name in self.standard_library
    
    def _read_file_with_fallback_encoding(self, file_path: str) -> str:
        """Read a file with fallback encodings if UTF-8 fails."""
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error opening {file_path}: {str(e)}")
                return ""
        
        # If all encodings fail, use binary mode and try to decode
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # Check for UTF-16 BOM
                if content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff'):
                    return content.decode('utf-16')
                # Try to decode as latin1 which rarely fails
                return content.decode('latin1')
        except Exception as e:
            self.encoding_errors += 1
            print(f"Failed to decode {file_path} with any encoding: {str(e)}")
            return ""
    
    def _extract_imports_with_ast(self, file_path: str) -> Set[str]:
        """Extract imports from a Python file using the AST module."""
        imports = set()
        
        try:
            content = self._read_file_with_fallback_encoding(file_path)
            if not content:
                return imports
            
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    # Handle regular imports: import X
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.add(self._extract_package_name(name.name))
                    
                    # Handle from imports: from X import Y
                    elif isinstance(node, ast.ImportFrom) and node.module is not None:
                        imports.add(self._extract_package_name(node.module))
            except SyntaxError:
                print(f"Syntax error in {file_path}, skipping...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
        
        return imports
    
    def _extract_imports_with_regex(self, file_path: str) -> Set[str]:
        """Extract imports from a Python file using regex (fallback method)."""
        imports = set()
        
        try:
            content = self._read_file_with_fallback_encoding(file_path)
            if not content:
                return imports
            
            for line in content.split('\n'):
                # Regular imports: import X
                match = IMPORT_RE.match(line.strip())
                if match:
                    imports.add(self._extract_package_name(match.group(1)))
                    continue
                
                # From imports: from X import Y
                match = FROM_IMPORT_RE.match(line.strip())
                if match:
                    imports.add(self._extract_package_name(match.group(1)))
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
        
        return imports
    
    def _get_package_version(self, package_name: str) -> Optional[str]:
        """Get the installed version of a package."""
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None
    
    def analyze_file(self, file_path: str) -> None:
        """
        Analyze a single Python file for imports.
        
        Args:
            file_path: Path to the Python file to analyze
        """
        if not file_path.endswith('.py'):
            return
        
        # Try with AST first, fall back to regex if AST fails
        imports = self._extract_imports_with_ast(file_path)
        if not imports:
            imports = self._extract_imports_with_regex(file_path)
        
        # Filter out project modules and standard library
        external_imports = {
            package for package in imports
            if not self._is_project_module(package) and not self._is_standard_library(package)
        }
        
        # Update import statistics
        for package in external_imports:
            self.all_imports.add(package)
            self.import_counts[package] += 1
    
    def analyze_directory(self, dir_path: str = None) -> None:
        """
        Recursively analyze all Python files in a directory.
        
        Args:
            dir_path: Directory path to analyze. Defaults to project root.
        """
        if dir_path is None:
            dir_path = self.project_root
        else:
            dir_path = Path(dir_path)
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    self.analyze_file(os.path.join(root, file))
    
    def generate_requirements(self, output_file: str = "requirements.txt") -> None:
        """
        Generate a requirements.txt file based on the analysis.
        
        Args:
            output_file: Path to the output requirements file
        """
        # Sort packages by import frequency
        sorted_packages = sorted(
            [(pkg, count) for pkg, count in self.import_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Requirements for ForestGaps-DL\n")
            f.write("# Generated by dependency_analyzer.py\n\n")
            
            # Write base requirements from setup.py
            f.write("# Base dependencies\n")
            for pkg, version in self._get_setup_dependencies():
                f.write(f"{pkg}{version}\n")
            
            f.write("\n# Additional detected dependencies\n")
            for package, count in sorted_packages:
                # Skip if already in setup.py dependencies
                if any(package == base_pkg for base_pkg, _ in self._get_setup_dependencies()):
                    continue
                
                version = self._get_package_version(package)
                if version:
                    f.write(f"{package}>={version}  # Used {count} times\n")
                else:
                    f.write(f"{package}  # Used {count} times, version unknown\n")
        
        print(f"Requirements file generated at {output_file}")
    
    def _get_setup_dependencies(self) -> List[Tuple[str, str]]:
        """Extract dependencies from setup.py."""
        setup_file = self.project_root / "setup.py"
        dependencies = []
        
        if setup_file.exists():
            try:
                content = self._read_file_with_fallback_encoding(str(setup_file))
                if content:
                    # Look for install_requires section
                    match = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
                    if match:
                        for line in match.group(1).split("\n"):
                            # Extract package and version
                            pkg_match = re.search(r'"([a-zA-Z0-9_-]+)([>=<].+)?"', line)
                            if pkg_match:
                                name = pkg_match.group(1)
                                version = pkg_match.group(2) or ""
                                dependencies.append((name, version))
            except Exception as e:
                print(f"Error processing setup.py: {str(e)}")
        
        return dependencies
    
    def print_summary(self) -> None:
        """Print a summary of the dependency analysis."""
        print("\n--- Dependency Analysis Summary ---")
        print(f"Total external packages found: {len(self.all_imports)}")
        if self.encoding_errors > 0:
            print(f"Warning: {self.encoding_errors} files could not be analyzed due to encoding issues")
        
        print("\nTop 10 most used packages:")
        
        for package, count in sorted(self.import_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            version = self._get_package_version(package) or "unknown"
            print(f"  - {package} (version: {version}): {count} occurrences")
        
        print("\nBase dependencies from setup.py:")
        for pkg, version in self._get_setup_dependencies():
            print(f"  - {pkg}{version}")


if __name__ == "__main__":
    # Use the current directory as the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    analyzer = DependencyAnalyzer(project_root)
    analyzer.analyze_directory()
    analyzer.print_summary()
    analyzer.generate_requirements("requirements.txt")