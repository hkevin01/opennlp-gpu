#!/bin/bash
# Script to build and run the OpenNLP GPU demo

# Build the project with Maven
mvn clean compile

# Run the demo
mvn exec:java
