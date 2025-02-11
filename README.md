# Dhalia API Deployment

## Quick Deploy Steps:

1. Set up AWS credentials (one-time setup)
   ```bash
   aws configure
   # Enter AWS keys and choose ap-southeast-1 region
   ```

2. Install requirements (one-time setup)
   ```bash
   pip install awsebcli
   ```

3. Make deploy script executable
   ```bash
   chmod +x deploy.sh
   ```

4. Deploy
   ```bash
   ./deploy.sh
   ```
