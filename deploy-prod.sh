#!/bin/bash
set -e

aws configure set aws_access_key_id XXXXXXXXXXX  --profile pollenprod
aws configure set aws_secret_access_key XXXXXXXXX --profile pollenprod
aws configure set region ap-southeast-1 --profile pollenprod
aws configure set output json --profile pollenprod

BEANSTALK_ENV="lms-agent"
DEPLOY_VERSION="app-$(git rev-parse --short HEAD)-$(date +%Y%m%d_%H%M%S)"

eb status $BEANSTALK_ENV --profile pollenprod > /dev/null 2>&1 || { echo "Environment $BEANSTALK_ENV not found"; }

# Create or update the environment
if ! eb status ${BEANSTALK_ENV} --profile pollenprod &> /dev/null; then
    echo -e "${YELLOW}Creating production environment...${NC}"
    eb create ${BEANSTALK_ENV} \
        --instance_type t2.micro \
        --single \
        --profile pollenprod \
        --timeout 20
    exit 1
fi

eb deploy $BEANSTALK_ENV \
 --profile pollenprod \
 --label "$DEPLOY_VERSION" \
 --timeout 10 \
 --process \
 || { echo "Deployment failed"; exit 1; }

echo "Waiting 60 seconds for deployment to stabilize..."
sleep 60

eb health $BEANSTALK_ENV --profile pollenprod \ | grep -q "Ok" \
 && echo "Deployment verified" \
 || { echo "Health check failed"; exit 1; }
