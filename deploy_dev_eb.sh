cd .
pwd
echo "Building dalia-api:1.0"
#docker build -t dalia-api:1.0 .


echo "Copying DEV .elasticbeanstalk project root"
#cp -r deploy/dev/.ebextensions .

cp  deploy/dev/.elasticbeanstalk/config.yml .elasticbeanstalk/config.yml

#eb init

eb create sb-dalia-api-service --profile pollensandbox