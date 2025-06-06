// Remove lombok imports
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test class for demonstrating standard Java patterns instead of Lombok.
 */
public class LombokTester {
    private static final Logger logger = LoggerFactory.getLogger(LombokTester.class);
    
    private final String name;
    
    /**
     * Default constructor.
     */
    public LombokTester() {
        this.name = "Default Name";
    }
    
    /**
     * Constructor with name parameter.
     * 
     * @param name the name
     */
    public LombokTester(String name) {
        this.name = name;
    }
    
    /**
     * Get the name.
     * 
     * @return the name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Test logging.
     */
    public void testLogging() {
        logger.info("Testing logging with field name: {}", name);
    }
}
