#!/bin/bash
# Generate code map for OpenNLP GPU extension project

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_FILE="$PROJECT_ROOT/docs/development/code_map.md"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Generating code map..."

# Start the markdown file
cat > "$OUTPUT_FILE" << 'EOF'
# OpenNLP GPU Extension - Code Map

This document provides an overview of the project structure and key components.

## Project Structure

```
EOF

# Generate directory tree (excluding build artifacts and IDE files)
(cd "$PROJECT_ROOT" && find . -type d \
    -not -path './target' \
    -not -path './target/*' \
    -not -path './.vscode' \
    -not -path './.vscode/*' \
    -not -path './build' \
    -not -path './build/*' \
    -not -path './.git' \
    -not -path './.git/*' \
    | sort | sed 's|[^/]*/|  |g; s|^  ||' >> "$OUTPUT_FILE")

echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add Java source files section
echo "## Java Source Files" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

find "$PROJECT_ROOT/src" -name "*.java" -type f | sort | while read -r java_file; do
    rel_path="${java_file#$PROJECT_ROOT/}"
    package=$(grep -E "^package\s+" "$java_file" 2>/dev/null | head -1 | sed 's/package\s*//;s/;//' || echo "unknown")
    class_name=$(grep -E "(class|interface|enum)\s+\w+" "$java_file" 2>/dev/null | head -1 | sed -E 's/.*[^a-zA-Z](class|interface|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*).*/\2/' || basename "$java_file" .java)
    
    echo "### $rel_path" >> "$OUTPUT_FILE"
    echo "- **Package**: $package" >> "$OUTPUT_FILE"
    echo "- **Class**: $class_name" >> "$OUTPUT_FILE"
    
    # Extract brief description from first comment or class javadoc
    description=$(grep -E "^\s*\*|//.*" "$java_file" | head -3 | sed 's|^\s*\*\s*||;s|^\s*//\s*||' | grep -v "^$" | head -1 || echo "")
    if [ -n "$description" ]; then
        echo "- **Description**: $description" >> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
done

echo "Code map generated: $OUTPUT_FILE"
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