#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="ap-southeast-1"  # Singapore region
ENVIRONMENT="prod"           # Production environment
BRANCH="main"               # Main branch

# Check if we're on the main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo -e "${RED}Error: Please switch to main branch first${NC}"
    echo -e "Run: ${GREEN}git checkout main${NC}"
    exit 1
fi

# Check if EB CLI is installed
if ! command -v eb &> /dev/null; then
    echo -e "${RED}Error: Elastic Beanstalk CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Initialize Elastic Beanstalk if needed
if [ ! -d .elasticbeanstalk ]; then
    echo -e "${YELLOW}Initializing Elastic Beanstalk...${NC}"
    eb init dhalia-api \
        --platform docker \
        --region ${AWS_REGION}
fi

# Build and tag Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t dhalia-api .

# Create or update the environment
if ! eb status ${ENVIRONMENT} &> /dev/null; then
    echo -e "${YELLOW}Creating production environment...${NC}"
    eb create ${ENVIRONMENT} \
        --instance_type t2.micro \
        --single \
        --cname dhalia-api-${ENVIRONMENT} \
        --timeout 20
else
    echo -e "${YELLOW}Deploying to production...${NC}"
    eb deploy ${ENVIRONMENT}
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
