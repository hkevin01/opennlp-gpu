# Lombok and H2 Database

This document explains two important tools used in the CLR WebUI backend: Lombok for reducing boilerplate code and H2 for testing.

## Project Lombok

### What is Lombok?

Project Lombok is a Java library that automatically plugs into your editor and build tools to reduce boilerplate code through annotations. It helps write cleaner, more concise code by generating common code patterns during compilation.

### Key Lombok Annotations

| Annotation | Purpose | Example Use |
|------------|---------|-------------|
| `@Getter` / `@Setter` | Generates getters and setters for fields | `@Getter private String name;` |
| `@NoArgsConstructor` | Creates a constructor with no parameters | `@NoArgsConstructor public class User {}` |
| `@AllArgsConstructor` | Creates a constructor with parameters for all fields | `@AllArgsConstructor public class User {}` |
| `@Data` | Shortcut for `@ToString`, `@EqualsAndHashCode`, `@Getter`, `@Setter`, and `@RequiredArgsConstructor` | `@Data public class User {}` |
| `@Builder` | Implements the builder pattern | `@Builder public class User {}` |
| `@Slf4j` | Creates a logger field | `@Slf4j public class Service { // log.info("...") }` |

### Lombok Examples

#### Example 1: Entity Class with Lombok

Without Lombok, a simple entity class would be very verbose:

```java
// WITHOUT LOMBOK - Very verbose
public class CoastalData {
    private Long id;
    private String location;
    private Double erosionRate;
    private LocalDate measurementDate;
    
    // Default constructor
    public CoastalData() {
    }
    
    // All args constructor
    public CoastalData(Long id, String location, Double erosionRate, LocalDate measurementDate) {
        this.id = id;
        this.location = location;
        this.erosionRate = erosionRate;
        this.measurementDate = measurementDate;
    }
    
    // Getters and setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public String getLocation() {
        return location;
    }
    
    public void setLocation(String location) {
        this.location = location;
    }
    
    public Double getErosionRate() {
        return erosionRate;
    }
    
    public void setErosionRate(Double erosionRate) {
        this.erosionRate = erosionRate;
    }
    
    public LocalDate getMeasurementDate() {
        return measurementDate;
    }
    
    public void setMeasurementDate(LocalDate measurementDate) {
        this.measurementDate = measurementDate;
    }
    
    // toString, equals, hashCode methods...
    @Override
    public String toString() {
        return "CoastalData{" +
                "id=" + id +
                ", location='" + location + '\'' +
                ", erosionRate=" + erosionRate +
                ", measurementDate=" + measurementDate +
                '}';
    }
    
    // equals and hashCode methods...
}
```

With Lombok, the same class becomes:

```java
// WITH LOMBOK - Clean and concise
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import java.time.LocalDate;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class CoastalData {
    @Id
    @GeneratedValue
    private Long id;
    
    private String location;
    private Double erosionRate;
    private LocalDate measurementDate;
}
```

#### Example 2: Service with Lombok Logging

```java
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j  // Creates a logger field named 'log'
@RequiredArgsConstructor  // Creates constructor for final fields
public class PredictionService {
    
    private final CoastalDataRepository coastalDataRepository;
    
    public PredictionResult predictCoastalChange(String location, int years) {
        log.info("Predicting coastal change for location: {} over {} years", location, years);
        
        try {
            // Service logic
            var coastalData = coastalDataRepository.findByLocation(location);
            log.debug("Found {} data points for location {}", coastalData.size(), location);
            
            // Perform prediction calculations
            var result = new PredictionResult();
            // ...
            
            log.info("Prediction completed successfully for {}", location);
            return result;
        } catch (Exception e) {
            log.error("Error predicting coastal change: {}", e.getMessage(), e);
            throw e;
        }
    }
}
```

#### Example 3: DTO with Lombok Builder Pattern

```java
import lombok.Builder;
import lombok.Value;
import java.time.LocalDate;

@Value  // Creates immutable class (final fields, getters, toString, equals, hashCode)
@Builder  // Creates builder pattern implementation
public class PredictionRequestDTO {
    String location;
    Integer timeframeYears;
    Boolean includeHistoricalData;
    LocalDate startDate;
    
    // Example usage:
    /*
    var request = PredictionRequestDTO.builder()
        .location("Miami Beach")
        .timeframeYears(10)
        .includeHistoricalData(true)
        .startDate(LocalDate.now())
        .build();
    */
}
```

### Benefits of Using Lombok

- **Reduced boilerplate**: Write significantly less code for common patterns
- **Improved readability**: Focus on business logic rather than repetitive code
- **Fewer bugs**: Less manual code means fewer opportunities for errors
- **Better maintainability**: Changes to class structure require fewer code updates

### How Lombok Works in Our Project

In the CLR WebUI project, we use Lombok to:

1. Keep our JPA entity classes clean and focused on their structure
2. Simplify service implementations with logging
3. Create immutable DTOs with builders for type-safe construction

### Setup

Lombok requires:
1. The Lombok dependency in `build.gradle`:
   ```gradle
   implementation 'org.projectlombok:lombok:1.18.32'
   annotationProcessor 'org.projectlombok:lombok:1.18.32'
   ```
2. IDE plugin for proper code completion (IntelliJ IDEA or Eclipse)

## H2 Database

### What is H2?

H2 is a lightweight, in-memory relational database written in Java. It can run in embedded mode (within your application), in-memory mode (data is lost when the application shuts down), or server mode.

### Key Features

- **In-memory operation**: Perfect for testing as it doesn't require persistent storage
- **Compatibility**: Supports standard SQL and JDBC API
- **Small footprint**: ~2MB JAR file size
- **Speed**: Fast startup and query execution
- **Browser console**: Built-in web console for database management

### H2 Examples

#### Example 1: H2 Configuration for Testing

In `src/test/resources/application-test.properties`:

```properties
# H2 Database Configuration for Testing
spring.datasource.url=jdbc:h2:mem:testdb;DB_CLOSE_DELAY=-1
spring.datasource.driverClassName=org.h2.Driver
spring.datasource.username=sa
spring.datasource.password=

# JPA Configuration
spring.jpa.database-platform=org.hibernate.dialect.H2Dialect
spring.jpa.hibernate.ddl-auto=create-drop
spring.jpa.show-sql=true

# H2 Console Configuration (for debugging)
spring.h2.console.enabled=true
spring.h2.console.path=/h2-console
```

#### Example 2: Integration Test with H2

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.ActiveProfiles;
import java.time.LocalDate;
import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest  // Uses embedded H2 database by default
@ActiveProfiles("test")  // Activates application-test.properties
public class CoastalDataRepositoryTest {

    @Autowired
    private CoastalDataRepository repository;

    @Test
    public void shouldSaveAndRetrieveCoastalData() {
        // Given
        CoastalData data = new CoastalData();
        data.setLocation("Test Beach");
        data.setErosionRate(-0.5);
        data.setMeasurementDate(LocalDate.now());
        
        // When
        repository.save(data);
        
        // Then
        var result = repository.findByLocation("Test Beach");
        assertThat(result).isNotEmpty();
        assertThat(result.get(0).getErosionRate()).isEqualTo(-0.5);
    }
    
    @Test
    public void shouldFindHighErosionAreas() {
        // Given
        CoastalData highErosion = new CoastalData();
        highErosion.setLocation("Eroding Beach");
        highErosion.setErosionRate(-2.5);
        highErosion.setMeasurementDate(LocalDate.now());
        
        CoastalData lowErosion = new CoastalData();
        lowErosion.setLocation("Stable Beach");
        lowErosion.setErosionRate(-0.1);
        lowErosion.setMeasurementDate(LocalDate.now());
        
        repository.save(highErosion);
        repository.save(lowErosion);
        
        // When
        var results = repository.findHighErosionAreas(-1.0);
        
        // Then
        assertThat(results).hasSize(1);
        assertThat(results.get(0).getLocation()).isEqualTo("Eroding Beach");
    }
}
```

#### Example 3: Using H2 Console for Debugging

H2 provides a web console for viewing the database during development or testing. To use it:

1. Add these properties to your `application.properties` or `application-dev.properties`:
   ```properties
   spring.h2.console.enabled=true
   spring.h2.console.path=/h2-console
   ```

2. Start your application with the dev profile:
   ```
   ./gradlew bootRun --args='--spring.profiles.active=dev'
   ```

3. Open a browser and navigate to `http://localhost:8080/h2-console`

4. Enter the JDBC URL, username, and password from your properties file.

Here's a screenshot of how the H2 console looks:
