#!/bin/bash
set -e

ENV="prod"
DEPLOY_VERSION="app-$(git rev-parse --short HEAD)-$(date +%Y%m%d_%H%M%S)"

eb status $ENV > /dev/null 2>&1 || { echo "Environment $ENV not found"; exit 1; }

eb deploy $ENV \
 --label "$DEPLOY_VERSION" \
 --timeout 10 \
 --process \
 || { echo "Deployment failed"; exit 1; }

echo "Waiting 60 seconds for deployment to stabilize..."
sleep 60

eb health $ENV | grep -q "Ok" \
 && echo "Deployment verified" \
 || { echo "Health check failed"; exit 1; }
