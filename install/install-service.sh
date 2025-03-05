sudo cp /home/ect-user/jina-embed-api/install/embed-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable embed-api
sudo systemctl start embed-api