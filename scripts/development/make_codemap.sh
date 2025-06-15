#!/usr/bin/env python3
# filepath: /home/kevin/Projects/opennlp-gpu/scripts/generate_code_map.py

import os
import json
import re
from pathlib import Path

def analyze_java_file(file_path):
    """Extract information from a Java file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract package
    package_match = re.search(r'package\s+([\w.]+);', content)
    package = package_match.group(1) if package_match else "unknown"
    
    # Extract class/interface name
    class_match = re.search(r'(public|private|protected)?\s+(class|interface|enum)\s+(\w+)', content)
    class_name = class_match.group(3) if class_match else os.path.basename(file_path).replace('.java', '')
    type_kind = class_match.group(2) if class_match else "unknown"
    
    # Determine if this is an implementation or interface
    is_interface = "interface" in content and "public interface" in content
    
    # Extract imports to determine dependencies
    imports = re.findall(r'import\s+([\w.]+);', content)
    
    # Check for annotations
    annotations = re.findall(r'@(\w+)', content)
    
    # Look for extends/implements
    extends_match = re.search(r'extends\s+([\w.]+)', content)
    extends = extends_match.group(1) if extends_match else None
    
    implements_match = re.search(r'implements\s+([\w.,\s]+)(?:\{|extends)', content)
    implements = [i.strip() for i in implements_match.group(1).split(',')] if implements_match else []
    
    # Try to determine purpose
    purpose = ""
    javadoc_match = re.search(r'/\*\*([\s\S]*?)\*/', content)
    if javadoc_match:
        # Extract from Javadoc
        javadoc = javadoc_match.group(1)
        lines = [line.strip().replace('*', '').strip() for line in javadoc.split('\n')]
        purpose = ' '.join([line for line in lines if line and not line.startswith('@')])
    
    if not purpose and "ComputeProvider" in content:
        purpose = "GPU compute provider implementation"
    elif not purpose and "Matrix" in class_name:
        purpose = "Matrix operations implementation"
    elif not purpose and "Util" in class_name:
        purpose = "Utility class for " + class_name.replace("Util", "")
    
    return {
        "name": class_name,
        "type": type_kind,
        "package": package,
        "path": str(file_path),
        "is_interface": is_interface,
        "imports": imports,
        "annotations": annotations,
        "extends": extends,
        "implements": implements,
        "purpose": purpose,
    }

def generate_code_map(root_dir):
    """Generate a code map for the project."""
    root = Path(root_dir)
    src_dir = root / "src" / "main" / "java"
    
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return {}
    
    code_map = {
        "project_name": "OpenNLP GPU",
        "description": "GPU acceleration for Apache OpenNLP using JOCL",
        "structure": {
            "components": [],
            "interfaces": [],
            "utilities": [],
        },
        "dependencies": {
            "external": [
                {"name": "JOCL", "purpose": "Java bindings for OpenCL"},
                {"name": "SLF4J", "purpose": "Logging abstraction"},
                {"name": "Lombok", "purpose": "Boilerplate code reduction"}
            ],
            "internal": []
        }
    }
    
    # Find all Java files
    java_files = list(src_dir.glob("**/*.java"))
    print(f"Found {len(java_files)} Java files")
    
    # Analyze each file
    components = []
    for java_file in java_files:
        try:
            file_info = analyze_java_file(java_file)
            if file_info["is_interface"]:
                code_map["structure"]["interfaces"].append(file_info)
            elif "Util" in file_info["name"]:
                code_map["structure"]["utilities"].append(file_info)
            else:
                code_map["structure"]["components"].append(file_info)
        except Exception as e:
            print(f"Error analyzing {java_file}: {e}")
    
    # Build internal dependencies
    for component in code_map["structure"]["components"]:
        for interface in code_map["structure"]["interfaces"]:
            if interface["name"] in component.get("implements", []):
                code_map["dependencies"]["internal"].append({
                    "from": component["name"],
                    "to": interface["name"],
                    "type": "implements"
                })
    
    # Add package structure
    packages = {}
    for item_type in ["components", "interfaces", "utilities"]:
        for item in code_map["structure"][item_type]:
            pkg = item["package"]
            if pkg not in packages:
                packages[pkg] = {"name": pkg, "items": []}
            packages[pkg]["items"].append(item["name"])
    
    code_map["packages"] = list(packages.values())
    
    return code_map

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_map = generate_code_map(project_root)
    
    output_file = os.path.join(project_root, "code_map.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(code_map, f, indent=2)
    
    print(f"Code map generated at {output_file}")
    
    # Print summary
    print("\nProject Summary:")
    print(f"Components: {len(code_map['structure']['components'])}")
    print(f"Interfaces: {len(code_map['structure']['interfaces'])}")
    print(f"Utilities: {len(code_map['structure']['utilities'])}")
    print(f"Packages: {len(code_map['packages'])}")