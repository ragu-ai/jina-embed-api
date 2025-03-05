sudo cp /home/ec2-user/jina-embed-api/install/embed-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable embed-api
sudo systemctl start embed-api