# Docker Multi-Platform Testing for OpenNLP GPU

This directory contains Docker configurations for testing the OpenNLP GPU project across multiple Linux distributions and environments.

## Supported Test Environments

| Environment     | Base Image                      | Package Manager | Description          |
| --------------- | ------------------------------- | --------------- | -------------------- |
| Ubuntu 22.04    | `ubuntu:22.04`                  | apt             | Latest Ubuntu LTS    |
| Ubuntu 20.04    | `ubuntu:20.04`                  | apt             | Previous Ubuntu LTS  |
| CentOS 8 Stream | `quay.io/centos/centos:stream8` | dnf             | Enterprise Linux     |
| Fedora 38       | `fedora:38`                     | dnf             | Latest Fedora        |
| Alpine Linux    | `alpine:3.18`                   | apk             | Minimal container OS |
| Amazon Linux 2  | `amazonlinux:2`                 | yum             | AWS optimized        |
| Debian 11       | `debian:11`                     | apt             | Stable Debian        |

## Quick Start

### Prerequisites

- Docker installed and running
- Docker Compose installed
- At least 4GB of available disk space

### Run All Tests

```bash
# From the project root directory
./scripts/run_docker_tests.sh
```

This will:
1. Build Docker images for all test environments
2. Run compatibility tests in each environment
3. Collect test results and generate a summary report
4. Clean up Docker resources

### Run Individual Environment Tests

```bash
# Test specific environment
cd docker/
docker-compose run --rm test-ubuntu22
docker-compose run --rm test-fedora38
docker-compose run --rm test-alpine
```

### Build Specific Images

```bash
# Build Ubuntu 22.04 test image
docker build -f docker/test-ubuntu22.04.Dockerfile -t opennlp-gpu-test-ubuntu22 .

# Build Alpine test image  
docker build -f docker/test-alpine.Dockerfile -t opennlp-gpu-test-alpine .
```

## Test Results

Test results are stored in:
- `test-output/docker-tests/` - Summary reports and logs
- `test-output/cross_platform_test_*.log` - Individual environment test logs

## What Gets Tested

Each Docker environment tests:

### Platform Detection
- Operating system identification
- Architecture detection  
- Distribution recognition
- Package manager detection

### Cross-Platform Utilities
- CPU core count detection
- Memory size detection
- Path separator handling
- Temporary directory access
- Home directory detection

### Script Compatibility
- Script syntax validation
- Executable permissions
- Error handling
- Cross-platform command usage

### Environment Setup
- Java installation and detection
- Maven installation and detection
- GPU detection capabilities
- Network connectivity
- File system permissions

### Project Functionality
- Maven compilation
- Basic script execution
- Error handling and fallbacks

## Customizing Tests

### Adding New Environments

1. Create a new Dockerfile in the `docker/` directory:
   ```dockerfile
   # Example: test-rockylinux9.Dockerfile
   FROM rockylinux:9
   # ... installation steps
   ```

2. Add the service to `docker-compose.yml`:
   ```yaml
   test-rockylinux9:
     build:
       context: ..
       dockerfile: docker/test-rockylinux9.Dockerfile
   ```

3. Update the environments list in `run_docker_tests.sh`:
   ```bash
   ENVIRONMENTS=(
       # ... existing environments
       "rockylinux9:Rocky Linux 9"
   )
   ```

### Modifying Test Behavior

Environment variables can customize test behavior:

- `CI=true` - Enables CI mode (non-interactive)
- `DOCKER_ENV=<name>` - Sets environment identifier
- `SKIP_NETWORK_TESTS=true` - Skips network connectivity tests
- `VERBOSE=true` - Enables verbose logging

## Troubleshooting

### Common Issues

**Docker daemon not running:**
```bash
sudo systemctl start docker
```

**Permission denied:**
```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

**Out of disk space:**
```bash
docker system prune -a
```

**Build failures:**
- Check individual Dockerfiles for package availability
- Verify base image versions are still supported
- Check network connectivity for package downloads

### Debugging Individual Environments

Run with shell access for debugging:
```bash
docker-compose run --rm --entrypoint /bin/bash test-ubuntu22
```

View build logs:
```bash
docker-compose build --no-cache test-ubuntu22
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Multi-Platform Docker Tests
on: [push, pull_request]

jobs:
  docker-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Docker tests
        run: ./scripts/run_docker_tests.sh
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: docker-test-results
          path: test-output/docker-tests/
```

### GitLab CI Example

```yaml
docker-tests:
  image: docker:20.10.16
  services:
    - docker:20.10.16-dind
  script:
    - ./scripts/run_docker_tests.sh
  artifacts:
    paths:
      - test-output/docker-tests/
    when: always
```

## Performance Considerations

- **Parallel execution**: Docker Compose runs services in parallel by default
- **Image caching**: Docker layer caching speeds up subsequent builds
- **Resource limits**: Each container uses ~1-2GB RAM during testing
- **Network usage**: Initial runs download base images (~200MB each)

## Security Notes

- All containers run as non-root where possible
- No sensitive data is included in images
- Images are built fresh for each test run
- Containers are automatically cleaned up after testing

## Windows Container Testing

### Requirements for Windows Containers

- **Windows Host:** Windows 10/11 Pro/Enterprise/Education or Windows Server 2016/2019/2022
- **Hyper-V:** Must be enabled in Windows Features
- **Docker Desktop:** Installed and configured for Windows containers
- **Disk Space:** 4-8 GB additional space for Windows base images

### Running Windows Tests

#### Method 1: Using Individual Script
```bash
# Run Windows-specific Docker tests
./scripts/test_windows_docker.sh
```

#### Method 2: Using Docker Compose
```bash
# Run Windows containers with profiles
cd docker
docker-compose --profile windows up --build
```

#### Method 3: Using Universal Test Runner
```bash
# Run all tests including Windows (when available)
./scripts/test_universal_compatibility.sh

# Run only Docker tests (includes Windows if available)
./scripts/test_universal_compatibility.sh --docker-only
```

### Switching to Windows Containers

1. Right-click Docker Desktop in system tray
2. Select "Switch to Windows containers..."
3. Wait for Docker to restart
4. Run tests: `./scripts/test_windows_docker.sh`

### Troubleshooting Windows Containers

**Common Issues:**

1. **"Docker is not running"**
   - Start Docker Desktop
   - Wait for it to fully initialize

2. **"Linux containers detected"**
   - Switch Docker Desktop to Windows containers mode
   - Restart Docker service

3. **"Hyper-V not enabled"**
   - Enable Hyper-V in Windows Features
   - Restart computer

4. **"Insufficient disk space"**
   - Free up at least 8GB disk space
   - Clean up unused Docker images: `docker system prune -a`

5. **Build timeouts**
   - Increase Docker build timeout settings
   - Check internet connection for downloading base images

### Windows Container Features

The Windows containers test:
- **Java 17 installation** via Chocolatey
- **Maven setup** and dependency resolution
- **Cross-platform script execution** in Windows environment
- **GPU detection capabilities** (when available)
- **Build and compilation** testing
- **Basic functionality** verification

### Performance Notes

- Windows containers are larger than Linux containers (4-8GB vs ~500MB)
- Initial builds take longer due to Windows base image downloads
- Runtime performance is comparable to native Windows applications

## Contributing

When adding new test environments:

1. Follow the existing naming convention: `test-<distro><version>.Dockerfile`
2. Include all required tools for compatibility testing
3. Set appropriate environment variables
4. Test the new environment locally before submitting
5. Update this README with the new environment details
