# OpenNLP GPU Extension - Installation Guide

## Quick Start (Maven)

Add this dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-gpu</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Quick Start (Gradle)

Add this to your `build.gradle`:

```gradle
implementation 'org.apache.opennlp:opennlp-gpu:1.0.0'
```

## Usage Example

```java
import org.apache.opennlp.gpu.integration.GpuModelFactory;
import opennlp.tools.ml.model.MaxentModel;

// Drop-in replacement for standard OpenNLP models
MaxentModel model = GpuModelFactory.createMaxentModel(existingModel);

// Automatic GPU acceleration when available, CPU fallback otherwise
double[] probabilities = model.eval(context);
```

## System Requirements

- **Java 11+**
- **Maven 3.6+** or **Gradle 6.0+**
- **GPU (Optional)**: CUDA, ROCm, or OpenCL compatible
- **OS**: Linux, Windows, macOS

## GPU Setup (Optional)

### NVIDIA CUDA
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvidia-smi
```

### AMD ROCm (Linux)
```bash
# Ubuntu/Debian
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms

# Verify installation
rocm-smi
```

### Intel OpenCL
```bash
# Ubuntu/Debian
sudo apt install intel-opencl-icd

# Verify installation
clinfo
```

## Verification

Test your installation:

```java
import org.apache.opennlp.gpu.integration.GpuModelFactory;

public class TestGpuIntegration {
    public static void main(String[] args) {
        System.out.println("GPU Available: " + GpuModelFactory.isGpuAvailable());
        System.out.println("GPU Info: " + GpuModelFactory.getGpuInfo());
    }
}
```

## Performance

Expected performance improvements:
- **Text Classification**: 3-5x faster
- **Named Entity Recognition**: 4-6x faster  
- **Feature Extraction**: 5-10x faster
- **Large Models**: Up to 15x faster

## Documentation

- [API Documentation](docs/api/README.md)
- [Performance Benchmarks](docs/performance/README.md)
- [GPU Configuration Guide](docs/setup/gpu_prerequisites_guide.md)
- [Migration Guide](docs/guides/migration_guide_reality_check.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/apache/opennlp-gpu/issues)
- **Documentation**: [Project Wiki](https://github.com/apache/opennlp-gpu/wiki)
- **Community**: [Apache OpenNLP Mailing List](https://opennlp.apache.org/mailing-lists.html)
