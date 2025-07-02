# Run

```terminal
./requirements/setup_venv.sh mistral
source ~/python_venvs/mistral/bin/activate
cd mistral_chat

# (OPTIONAL) Get index and 
(mistral) python3 build_index.py

# launch server with fastapi
(mistral) uvicorn main:app --host 0.0.0.0 --port 8000

# open browser
http://0.0.0.0:8000/
```

# Example

In the two example below we observe how the model fails to preserve context already in the second question, namely *How did it end?*, while the solution with LangChain incorporated (in particular utilizing `ConversationBufferMemory` from `langchain.memory`) seems to succeed in preserving context by taking into consideration earlier prompts.

| Without LangChain                                            | With LangChain                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="img/chat_without_langchain.png" alt="drawing" width="450"/> | <img src="img/chat_with_langchain.png" alt="drawing" width="450"/> |

# Docker Cheatsheet

## 1. Build the image

From your project root (where `Dockerfile` lives):

```bash
# Build and tag
docker build -t mistral-chat:latest .
```

## 2. List images

See what’s on your host:

```bash
docker images
```

## 3. Run a container

Start your app (detached, publish port 8000):

```bash
docker run -d \
  --name mistral-server \
  --gpus all \            # omit if CPU-only
  -p 8000:8000 \
  mistral-chat:latest
```

- `-d` = detached (runs in background)
- `--name` = give it an easy handle
- `--gpus all` = expose all GPUs (needs NVIDIA Container Toolkit)
- `-p host_port:container_port` = map ports

Check it’s up at `http://<host-ip>:8000`.

## 4. See running containers

```bash
docker ps
```

Add `-a` to include stopped ones:

```bash
docker ps -a
```

## 5. Stop / start / restart

```bash
docker stop mistral-server      # graceful shutdown
docker start mistral-server     # bring it back up
docker restart mistral-server   # stop + start in one go
```

You can also use the container **ID** instead of name.

## 6. View logs

```bash
docker logs -f mistral-server   # -f to “follow” new output
```

## 7. Execute a shell inside

```bash
docker exec -it mistral-server /bin/bash
```

Handy for inspecting files or testing commands in the live container.

## 8. Remove containers and images

```bash
# Remove a stopped container
docker rm mistral-server

# Remove the image
docker rmi mistral-chat:latest
```

If you need to force-remove a running container, add `-f` to `docker rm`.

## 9. Clean up unused resources

```bash
# Remove all stopped containers, dangling images, unused networks
docker system prune
```

Add `-a` to also remove unused images:

```bash
docker system prune -a
```

## 10. (Optional) Docker Compose

If you create a `docker-compose.yml`, you can:

```bash
# Start services in foreground
docker-compose up

# Start in background
docker-compose up -d

# Stop all services
docker-compose down
```

### Quick reference

| Command                         | Action                                                |
| ------------------------------- | ----------------------------------------------------- |
| `docker build -t name:tag .`    | Build image                                           |
| `docker images`                 | List images                                           |
| `docker run -d --name c -p 8:8` | Run container (detached)                              |
| `docker ps` / `docker ps -a`    | List running / all containers                         |
| `docker stop c`                 | Stop a container                                      |
| `docker start c`                | Start a stopped container                             |
| `docker restart c`              | Restart container                                     |
| `docker logs -f c`              | Tail container logs                                   |
| `docker exec -it c /bin/bash`   | Shell into container                                  |
| `docker rm c`                   | Remove a container                                    |
| `docker rmi name:tag`           | Remove an image                                       |
| `docker system prune -a`        | Remove all unused data (containers, images, networks) |



# Docker setup

1. Make sure you install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. On your terminal

   ```terminal
   sudo touch /etc/docker/daemon.json
   sudo gedit /etc/docker/daemon.json
   ```

   and populate it with this

   ```json
   {
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

3. Restart Docker

   ```terminal
   sudo systemctl restart docker
   ```

4. Confirm that Docker daemon is wired up to use the NVIDIA runtime

   ```terminal
   docker info | grep -E "Runtimes|Default Runtime"
    Runtimes: io.containerd.runc.v2 nvidia runc
    Default Runtime: nvidia
   ```

5. Test GPU access

   ```terminal
    docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
    
    ==========
   == CUDA ==
   ==========
   
   CUDA Version 12.1.1
   
   Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   
   This container image and its contents are governed by the NVIDIA Deep Learning Container License.
   By pulling and using the container, you accept the terms and conditions of this license:
   https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
   
   A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
   
   Fri Jun 27 09:45:07 2025       
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
   |-----------------------------------------+------------------------+----------------------+
   | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
   |                                         |                        |               MIG M. |
   |=========================================+========================+======================|
   |   0  NVIDIA GeForce RTX 4050 ...    Off |   00000000:01:00.0 Off |                  N/A |
   | N/A   44C    P8              4W /   80W |      18MiB /   6141MiB |      0%      Default |
   |                                         |                        |                  N/A |
   +-----------------------------------------+------------------------+----------------------+
                                                                                            
   +-----------------------------------------------------------------------------------------+
   | Processes:                                                                              |
   |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
   |        ID   ID                                                               Usage      |
   |=========================================================================================|
   +-----------------------------------------------------------------------------------------+
   ```

   
