# Apache OpenNLP Fork Instructions and Contribution Protocol

## Overview
This guide provides step-by-step instructions for properly forking Apache OpenNLP and contributing your GPU acceleration features following Apache protocols.

## ⚠️ Important: Contribution Protocol Order

**DO NOT fork immediately!** Follow this exact order:

### 1. Community Engagement FIRST ✅
- Email the community proposal
- Get feedback and approval
- Create JIRA issue
- THEN fork and develop

## Phase 1: Community Engagement (REQUIRED FIRST)

### Step 1.1: Subscribe to OpenNLP Mailing Lists
```bash
# Subscribe to developer mailing list
echo "subscribe" | mail dev-subscribe@opennlp.apache.org

# Subscribe to user mailing list (optional but recommended)
echo "subscribe" | mail users-subscribe@opennlp.apache.org
```

**Alternative**: Visit https://opennlp.apache.org/mailing-lists.html and subscribe via web interface

### Step 1.2: Send Community Proposal
1. **Use the prepared email**: `docs/apache_proposal_email_final.md`
2. **Customize with your information**:
   - Add your name, email, GitHub profile
   - Add your current repository URL
   - Include any specific performance benchmarks
3. **Send to**: dev@opennlp.apache.org
4. **Subject**: [PROPOSAL] GPU Acceleration Framework for Apache OpenNLP

### Step 1.3: Engage in Community Discussion
- **Respond promptly** to community questions
- **Provide additional details** as requested
- **Be open to feedback** and architectural suggestions
- **Build consensus** before proceeding to code

### Step 1.4: Create JIRA Issue (After Community Approval)
1. **Visit**: https://issues.apache.org/jira/projects/OPENNLP
2. **Create new issue**:
   - **Issue Type**: New Feature
   - **Summary**: "GPU Acceleration Support for OpenNLP"
   - **Description**: Use template below
   - **Components**: Tools, Models (as applicable)
   - **Labels**: gpu, acceleration, performance, enhancement

#### JIRA Issue Template:
```
## Summary
Add comprehensive GPU acceleration support to Apache OpenNLP

## Description
This feature request proposes adding GPU acceleration capabilities to OpenNLP, providing 3-50x performance improvements for various NLP operations while maintaining 100% backward compatibility.

## Benefits
- 3-50x performance improvements across tokenization, feature extraction, model training
- Zero breaking changes - existing code works unchanged
- Enterprise-ready with production monitoring and optimization
- Hardware agnostic (NVIDIA, AMD, Intel GPUs)

## Implementation Approach
- Optional GPU acceleration module
- Automatic CPU fallback when GPU unavailable
- Minimal API surface changes
- Comprehensive test coverage

## Performance Improvements
- Tokenization: 3-5x faster
- Feature Extraction: 5-8x faster  
- Model Training: 8-15x faster
- Batch Processing: 10-25x faster
- Neural Networks: 15-50x faster

## Community Discussion
Mailing list thread: [Link to your proposal thread]

## Implementation Status
Complete implementation available at: [Your repository URL]
Ready for integration following community approval
```

## Phase 2: Technical Preparation (After Community Approval)

### Step 2.1: Study Apache OpenNLP Architecture
```bash
# Clone OpenNLP to study structure
git clone https://github.com/apache/opennlp.git opennlp-study
cd opennlp-study

# Study the codebase structure
find . -name "*.java" | head -20
ls -la opennlp-tools/src/main/java/opennlp/tools/

# Understand build system
cat pom.xml
ls opennlp-*/pom.xml
```

### Step 2.2: Understand Apache Development Standards
1. **Read Apache OpenNLP Guidelines**:
   - https://opennlp.apache.org/docs/
   - https://github.com/apache/opennlp/blob/main/CONTRIBUTING.md
   
2. **Study Code Style**:
   - Java coding conventions
   - JavaDoc standards
   - Test patterns
   - Package organization

3. **Review Recent Contributions**:
   - Look at recent pull requests
   - Understand review process
   - See what gets accepted/rejected

## Phase 3: Fork and Development (After Community Approval)

### Step 3.1: Proper Fork Creation
```bash
# 1. Fork on GitHub
# Go to https://github.com/apache/opennlp
# Click "Fork" button (top right)
# This creates YourUsername/opennlp

# 2. Clone your fork
git clone https://github.com/YourUsername/opennlp.git
cd opennlp

# 3. Add upstream remote
git remote add upstream https://github.com/apache/opennlp.git

# 4. Verify remotes
git remote -v
# Should show:
# origin    https://github.com/YourUsername/opennlp.git (fetch)
# origin    https://github.com/YourUsername/opennlp.git (push)  
# upstream  https://github.com/apache/opennlp.git (fetch)
# upstream  https://github.com/apache/opennlp.git (push)
```

### Step 3.2: Create Feature Branch
```bash
# 1. Ensure you're on main branch
git checkout main

# 2. Pull latest changes
git fetch upstream
git rebase upstream/main

# 3. Create feature branch (use JIRA issue number)
git checkout -b OPENNLP-XXXX-gpu-acceleration
# Replace XXXX with your JIRA issue number
```

### Step 3.3: Plan Integration Architecture

#### Recommended Module Structure:
```
opennlp/
├── opennlp-tools/              # Core OpenNLP (existing)
├── opennlp-gpu/               # NEW: GPU acceleration module
│   ├── src/main/java/
│   │   └── opennlp/tools/gpu/
│   │       ├── common/         # Core GPU interfaces
│   │       ├── compute/        # GPU compute implementations  
│   │       ├── features/       # GPU feature extraction
│   │       ├── ml/            # GPU machine learning
│   │       └── integration/    # OpenNLP integration
│   ├── src/test/java/         # Comprehensive tests
│   └── pom.xml               # Module POM
├── opennlp-examples/          # Updated examples (existing)
└── pom.xml                   # Updated parent POM
```

#### Integration Points:
1. **Minimal Changes to Core**: Keep opennlp-tools unchanged
2. **Optional Dependency**: GPU module is optional
3. **Service Provider Interface**: Use SPI for GPU providers
4. **Configuration**: Standard OpenNLP configuration patterns

### Step 3.4: Code Migration Strategy

#### Phase 1: Core Infrastructure
```bash
# Create new module
mkdir opennlp-gpu
cd opennlp-gpu

# Set up Maven structure
mkdir -p src/main/java/opennlp/tools/gpu
mkdir -p src/test/java/opennlp/tools/gpu
mkdir -p src/main/resources
mkdir -p src/test/resources

# Create module POM
cat > pom.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.apache.opennlp</groupId>
        <artifactId>opennlp</artifactId>
        <version>2.4.0-SNAPSHOT</version>
    </parent>
    
    <artifactId>opennlp-gpu</artifactId>
    <name>Apache OpenNLP GPU Acceleration</name>
    <description>GPU acceleration support for Apache OpenNLP</description>
    
    <dependencies>
        <dependency>
            <groupId>org.apache.opennlp</groupId>
            <artifactId>opennlp-tools</artifactId>
            <version>${project.version}</version>
        </dependency>
        
        <!-- GPU runtime dependencies -->
        <dependency>
            <groupId>org.aparapi</groupId>
            <artifactId>aparapi</artifactId>
            <version>3.0.0</version>
            <optional>true</optional>
        </dependency>
    </dependencies>
</project>
EOF
```

#### Phase 2: Migrate Core Classes
1. **Copy your GPU classes** to new structure
2. **Update package names** to `opennlp.tools.gpu.*`
3. **Follow OpenNLP naming conventions**
4. **Add proper Apache license headers**

#### Phase 3: Integration Layer
1. **Create SPI interfaces** for GPU providers
2. **Implement OpenNLP integration points**
3. **Add configuration support**
4. **Create factory classes**

### Step 3.5: Apache License Headers
Add to all Java files:
```java
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

## Phase 4: Development Best Practices

### Step 4.1: Regular Sync with Upstream
```bash
# Weekly sync with upstream
git fetch upstream
git rebase upstream/main

# Resolve conflicts if any
# Push to your fork
git push origin OPENNLP-XXXX-gpu-acceleration
```

### Step 4.2: Testing Strategy
1. **All existing tests must pass**
2. **New tests for GPU functionality**
3. **Integration tests with OpenNLP core**
4. **Performance regression tests**

### Step 4.3: Documentation Requirements
1. **JavaDoc for all public APIs**
2. **User guide updates**
3. **Performance benchmarking results**
4. **Migration examples**

## Phase 5: Submission Process

### Step 5.1: Pre-submission Checklist
- [ ] All tests pass
- [ ] Code follows Apache style
- [ ] Documentation complete
- [ ] No breaking changes
- [ ] Performance improvements documented
- [ ] JIRA issue updated

### Step 5.2: Create Pull Request
```bash
# Ensure branch is up to date
git fetch upstream
git rebase upstream/main

# Push final version
git push origin OPENNLP-XXXX-gpu-acceleration

# Create PR via GitHub:
# 1. Go to your fork on GitHub
# 2. Click "Compare & pull request"
# 3. Target: apache/opennlp main branch
# 4. Source: YourUsername/opennlp OPENNLP-XXXX-gpu-acceleration
```

#### Pull Request Template:
```markdown
## Summary
Implements GPU acceleration support for Apache OpenNLP (OPENNLP-XXXX)

## Changes
- Added opennlp-gpu module with comprehensive GPU acceleration
- Zero breaking changes to existing APIs
- Optional dependency with automatic CPU fallback
- Enterprise-ready with monitoring and optimization

## Performance Improvements
- Tokenization: 3-5x faster
- Feature extraction: 5-8x faster
- Model training: 8-15x faster
- Batch processing: 10-25x faster

## Testing
- All existing tests pass
- 95%+ test coverage for new functionality
- Integration tests with OpenNLP core
- Performance regression tests included

## Documentation
- Complete JavaDoc for all public APIs
- User guide updates with examples
- Performance benchmarking results
- Migration guide for existing projects

## JIRA
Closes OPENNLP-XXXX
```

### Step 5.3: Address Review Feedback
- **Respond promptly** to reviewer comments
- **Make requested changes** in separate commits
- **Explain design decisions** when needed
- **Update documentation** as required

## ✅ Summary Checklist

### Before You Start:
- [ ] Send community proposal email
- [ ] Get community feedback and approval
- [ ] Create JIRA issue
- [ ] Understand Apache development standards

### Development Phase:
- [ ] Fork apache/opennlp repository
- [ ] Create feature branch with JIRA number
- [ ] Migrate code following Apache patterns
- [ ] Add proper license headers
- [ ] Ensure all tests pass
- [ ] Add comprehensive documentation

### Submission Phase:
- [ ] Sync with latest upstream
- [ ] Create pull request with detailed description
- [ ] Link to JIRA issue
- [ ] Address review feedback promptly
- [ ] Maintain code until merged

## 🚨 Critical Success Factors

1. **Community First**: Always engage community before coding
2. **Quality Standards**: Meet Apache quality requirements
3. **Backward Compatibility**: Zero breaking changes
4. **Documentation**: Comprehensive docs and examples
5. **Long-term Commitment**: Be prepared for maintenance

## 📞 Getting Help

- **OpenNLP Dev List**: dev@opennlp.apache.org
- **OpenNLP JIRA**: https://issues.apache.org/jira/projects/OPENNLP
- **Apache Mentors**: Available for guidance on contribution process
- **Community Slack**: Join OpenNLP community channels

Remember: **Community engagement is more important than perfect code**. Start with the proposal email and build relationships first!
