git clone <URL_REPOSITORY>
cd tool_learn_aws

sudo systemctl enable --now docker
sudo make download-tts-model
mkdir -p data/voice
cp "/đường/dẫn/file-giọng.mp4" data/voice/
sudo make voice
sudo make up
sudo make logs
sudo make down
sudo make up
sudo docker compose build web
sudo docker compose up -d --force-recreate web worker

# tool_learn
