# Log Files

This directory contains log files generated during setup, build, and runtime operations.

## Files

- `setup.log` - Detailed setup process logs
- `setup-errors.log` - Setup error messages and diagnostics

## Log Types

### Setup Logs
- **`setup.log`** - Complete setup process log including:
  - System detection
  - Dependency installation
  - Build process
  - Verification steps
  - Performance information

- **`setup-errors.log`** - Error-specific logs including:
  - Build failures
  - Dependency issues
  - System compatibility problems
  - GPU driver issues

## Usage

### Viewing Logs
```bash
# View setup log
tail -f logs/setup.log

# View error log
cat logs/setup-errors.log

# Search for specific issues
grep -i "error\|fail" logs/setup.log
```

### Troubleshooting
When setup fails:
1. Check `setup-errors.log` for specific error messages
2. Review `setup.log` for context around failures
3. Use logs to identify system requirements not met

## Log Rotation

Logs are created fresh each time setup scripts run. Previous logs are automatically backed up with timestamps.
