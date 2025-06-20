# Java Environment Setup Guide

## Automatic Setup
Run the automatic setup script:
```bash
./scripts/fix_java_environment.sh
```

## Manual Verification
Check your Java environment:
```bash
./scripts/validate_java_environment.sh
```

## VSCode Integration
If VSCode shows Java version errors:
```bash
./scripts/setup_vscode_java.sh
# Then reload VSCode: Ctrl+Shift+P -> "Developer: Reload Window"
```

## Troubleshooting

### Issue: "Java version is older than Java 11"
**Solution**: 
```bash
./scripts/fix_java_environment.sh
# Reload VSCode
```

### Issue: Maven using wrong Java version
**Solution**:
```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
mvn clean compile
```

### Issue: VSCode XML server errors
**Solution**: VSCode settings are automatically configured to use Java 21 for XML processing.

## Environment Guard
The environment guard script runs automatically to ensure Java stays configured correctly:
```bash
./scripts/vscode_java_guard.sh
```
