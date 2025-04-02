# Changes from upstream at cvat-ai/cvat

- Added convenience compose files and envvars to more easily run annotation models (nuclio serverless functions) either locally or on a separate server
- Added deploy_cpu.ps1 and deploy_gpu.ps1 for easier Windows deployment of models
- Added support for running SAM2 (Credit to [hashJoe/cvat:feature/sam2](https://github.com/cvat-ai/cvat/pull/8610))

The following CVAT versions are available - see below for instructions on how to update to newer versions:
- [v2.32.0-sam2](https://github.com/dk-teknologisk-jahs/cvat/tree/v2.32.0-sam2)

## Running this fork of CVAT

Requirements:
- Docker & Docker Compose

```bash
# Clone our fork
git clone https://github.com/dk-teknologisk-jahs/cvat.git
cd cvat

# Switch to branch with changes
git switch v2.32.0-sam2

# Check .env and change any variables as necessary, such as CVAT_NUCLIO_HOST

# Build & Start CVAT using the provided compose files (should use compose.yaml by default, add -f compose.yaml if not)
docker compose up -d --build --force-recreate --renew-anon-volumes
```

## Commands used to create this fork (Bash)

```bash
# Clone our fork
git clone https://github.com/dk-teknologisk-jahs/cvat.git
cd cvat

# Add required remotes
git remote add upstream https://github.com/cvat-ai/cvat.git
git remote add hashJoe https://github.com/hashJoe/cvat.git

# Fetch everything
git fetch --all --tags

# Find latest stable version
git tag -l | sort -V
# Let's assume v2.32.0 is the latest stable

# Create a branch based on this stable version
git checkout -b v2.32.0-sam2 v2.32.0

# Merge the SAM2 feature
git merge hashJoe/feature/sam2

# Add necessary files to run CVAT with SAM2 and either local or remote nuclio server
cat <<'EOF' >compose.yaml
include:
  - path:
    - ./docker-compose.yml
    #- ./docker-compose.dev.yml # should not be necessary (warning: will expose stuff like the database on port 5432)
    - ./components/serverless/docker-compose.serverless.yml
    - ./compose.override.yaml
EOF
cat <<'EOF' >compose.override.yaml
services:
  cvat_server:
    environment:
      CVAT_SERVERLESS: 1
      CVAT_NUCLIO_HOST: ${CVAT_NUCLIO_HOST:-localhost}
      CVAT_NUCLIO_INVOKE_METHOD: 'dashboard'

  cvat_worker_annotation:
    environment:
      CVAT_NUCLIO_HOST: ${CVAT_NUCLIO_HOST:-localhost}
      CVAT_NUCLIO_INVOKE_METHOD: 'dashboard'
EOF
cat <<'EOF' >.env
CLIENT_PLUGINS=plugins/sam2
CVAT_HOST=tjorn.local # CVAT has some issues with 404 erros if we don't set this
CVAT_SERVERLESS=1
CVAT_NUCLIO_HOST=172.17.155.175 # set to localhost if using local nuclio
CVAT_NUCLIO_INVOKE_METHOD=dashboard
EOF
git add -f compose.yaml compose.override.yaml .env
git commit -m 'Added compose.yml, compose.override.yaml and .env'

# Add deploy_cpu.ps1 and deploy_gpu.ps1 for Windows deployment of models
cat <<'EOF' >serverless/deploy_cpu.ps1
# deploy_cpu.ps1 - Windows version of deploy_cpu.sh
# Sample commands to deploy nuclio functions on CPU

# Enable stopping on errors
$ErrorActionPreference = "Stop"

# Get the script directory
$SCRIPT_DIR = $PSScriptRoot
$FUNCTIONS_DIR = if ($args[0]) { $args[0] } else { $SCRIPT_DIR }

# Enable Docker BuildKit
$env:DOCKER_BUILDKIT = 1

# Build base OpenVINO image
docker build -t cvat.openvino.base "$SCRIPT_DIR\openvino\base"

# Create the CVAT project
nuctl create project cvat --platform local

# Find and deploy all function.yaml files
Get-ChildItem -Path $FUNCTIONS_DIR -Recurse -Filter "function.yaml" | ForEach-Object {
    $func_config = $_.FullName
    $func_root = Split-Path -Parent $func_config
    $func_parent_dir = Split-Path -Parent $func_root

    # Calculate relative path similar to Linux's realpath
    $func_rel_path = $func_parent_dir.Replace($SCRIPT_DIR, "").TrimStart("\")

    # Build Docker image if Dockerfile exists
    if (Test-Path -Path "$func_root\Dockerfile") {
        $docker_tag = "cvat." + ($func_rel_path -replace "\\", ".") + ".base"
        docker build -t $docker_tag $func_root
    }

    Write-Host "Deploying $func_rel_path function..."
    nuctl deploy --project-name cvat --path $func_root `
        --file $func_config --platform local `
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk `
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 `
        --platform-config '{\"attributes\": {\"network\": \"cvat_cvat\"}}'
}

# List deployed functions
nuctl get function --platform local
EOF
cat <<'EOF' >serverless/deploy_gpu.ps1
# deploy_gpu.ps1 - Windows version of deploy_gpu.sh
# Sample commands to deploy nuclio functions on GPU

# Enable stopping on errors
$ErrorActionPreference = "Stop"

# Get the script directory
$SCRIPT_DIR = $PSScriptRoot
$FUNCTIONS_DIR = if ($args[0]) { $args[0] } else { $SCRIPT_DIR }

# Create the CVAT project
nuctl create project cvat --platform local

# Find and deploy all function-gpu.yaml files
Get-ChildItem -Path $FUNCTIONS_DIR -Recurse -Filter "function-gpu.yaml" | ForEach-Object {
    $func_config = $_.FullName
    $func_root = Split-Path -Parent $func_config
    $func_parent_dir = Split-Path -Parent $func_root

    # Calculate relative path similar to Linux's realpath
    $func_rel_path = $func_parent_dir.Replace($SCRIPT_DIR, "").TrimStart("\")

    Write-Host "Deploying $func_rel_path function..."
    nuctl deploy --project-name cvat --path $func_root `
        --file $func_config --platform local `
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk `
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 `
        --platform-config '{\"attributes\": {\"network\": \"cvat_cvat\"}}'
}

# List deployed functions
nuctl get function --platform local
EOF
git add -f serverless/deploy_cpu.ps1 serverless/deploy_gpu.ps1
git commit -m 'Added deploy_cpu.ps1 and deploy_gpu.ps1 for Windows'

# Compatibility fix for yolov7
sed -i 's/baseImage: nvidia\/cuda:12\.6\.3-cudnn-runtime-ubuntu22\.04/baseImage: nvidia\/cuda:12\.4\.1-cudnn-runtime-ubuntu22\.04/g' serverless/onnx/WongKinYiu/yolov7/nuclio/function-gpu.yaml
git commit -m 'Downgrade cuda of yolov7 from 2.6.3 to 2.4.1 for better compatibility'

# Resolve any conflicts if necessary
# Then commit and push
git push -u origin v2.32.0-sam2
```

## Updating to a new CVAT version

When CVAT releases a new version (e.g. v2.45.0), you can either just merge the changes since the last stable version:

```bash
# If you have made changes to the .env etc, make sure you are on the previous stable
# version w. SAM2 (v2.32.0-sam2 in this case), so you can stash the local changes
git checkout v2.32.0-sam2 # replace with the branch you are using now
git stash push -u -m "local_changes"

# Create a new branch from the previous stable version w. SAM2 (v2.32.0-sam2 in this case)
git fetch upstream --tags
git checkout -b v2.45.0-sam2 v2.32.0-sam2 # replace with the new CVAT version you want to use and the branch you are using now

# Now merge the changes since previous stable version
git merge upstream/v2.45.0 # replace with the new CVAT version you want to use

# Pop the stashed changes if necessary
git stash apply stash^{/local_changes}

# Resolve conflicts, test, then optionally push
git push -u origin v2.45.0-sam2 # replace with the new CVAT version you want to use
```

Or alternatively, start from scratch, reapply the SAM2 changes and add all necessary files:

```bash
# If you have made changes to the .env etc, make sure you are on the previous stable
# version w. SAM2 (v2.32.0-sam2 in this case), so you can stash the local changes
git checkout v2.32.0-sam2 # replace with the branch you are using now
git stash push -u -m "local_changes"

# Create a new branch from the new stable version wo. SAM2 (v2.45.0 in this case)
git fetch upstream --tags
git checkout -b v2.45.0-sam2 upstream/v2.45.0 # replace with the new CVAT version you want to use and the branch you are using now

# Now merge the changes from the SAM2 feature branch
git merge hashJoe/feature/sam2

# Add necessary files to run CVAT with SAM2 and either local or remote nuclio server
cat <<'EOF' >compose.yaml
include:
  - path:
    - ./docker-compose.yml
    #- ./docker-compose.dev.yml # should not be necessary (warning: will expose stuff like the database on port 5432)
    - ./components/serverless/docker-compose.serverless.yml
    - ./compose.override.yaml
EOF
cat <<'EOF' >compose.override.yaml
services:
  cvat_server:
    environment:
      CVAT_SERVERLESS: 1
      CVAT_NUCLIO_HOST: ${CVAT_NUCLIO_HOST:-localhost}
      CVAT_NUCLIO_INVOKE_METHOD: 'dashboard'

  cvat_worker_annotation:
    environment:
      CVAT_NUCLIO_HOST: ${CVAT_NUCLIO_HOST:-localhost}
      CVAT_NUCLIO_INVOKE_METHOD: 'dashboard'
EOF
cat <<'EOF' >.env
CLIENT_PLUGINS=plugins/sam2
CVAT_HOST=tjorn.local # CVAT has some issues with 404 erros if we don't set this
CVAT_SERVERLESS=1
CVAT_NUCLIO_HOST=172.17.155.175 # set to localhost if using local nuclio
CVAT_NUCLIO_INVOKE_METHOD=dashboard
EOF
git add -f compose.yaml compose.override.yaml .env
git commit -m 'Added compose.yml, compose.override.yaml and .env'

# Add deploy_cpu.ps1 and deploy_gpu.ps1 for Windows deployment of models
cat <<'EOF' >serverless/deploy_cpu.ps1
# deploy_cpu.ps1 - Windows version of deploy_cpu.sh
# Sample commands to deploy nuclio functions on CPU

# Enable stopping on errors
$ErrorActionPreference = "Stop"

# Get the script directory
$SCRIPT_DIR = $PSScriptRoot
$FUNCTIONS_DIR = if ($args[0]) { $args[0] } else { $SCRIPT_DIR }

# Enable Docker BuildKit
$env:DOCKER_BUILDKIT = 1

# Build base OpenVINO image
docker build -t cvat.openvino.base "$SCRIPT_DIR\openvino\base"

# Create the CVAT project
nuctl create project cvat --platform local

# Find and deploy all function.yaml files
Get-ChildItem -Path $FUNCTIONS_DIR -Recurse -Filter "function.yaml" | ForEach-Object {
    $func_config = $_.FullName
    $func_root = Split-Path -Parent $func_config
    $func_parent_dir = Split-Path -Parent $func_root

    # Calculate relative path similar to Linux's realpath
    $func_rel_path = $func_parent_dir.Replace($SCRIPT_DIR, "").TrimStart("\")

    # Build Docker image if Dockerfile exists
    if (Test-Path -Path "$func_root\Dockerfile") {
        $docker_tag = "cvat." + ($func_rel_path -replace "\\", ".") + ".base"
        docker build -t $docker_tag $func_root
    }

    Write-Host "Deploying $func_rel_path function..."
    nuctl deploy --project-name cvat --path $func_root `
        --file $func_config --platform local `
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk `
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 `
        --platform-config '{\"attributes\": {\"network\": \"cvat_cvat\"}}'
}

# List deployed functions
nuctl get function --platform local
EOF
cat <<'EOF' >serverless/deploy_gpu.ps1
# deploy_gpu.ps1 - Windows version of deploy_gpu.sh
# Sample commands to deploy nuclio functions on GPU

# Enable stopping on errors
$ErrorActionPreference = "Stop"

# Get the script directory
$SCRIPT_DIR = $PSScriptRoot
$FUNCTIONS_DIR = if ($args[0]) { $args[0] } else { $SCRIPT_DIR }

# Create the CVAT project
nuctl create project cvat --platform local

# Find and deploy all function-gpu.yaml files
Get-ChildItem -Path $FUNCTIONS_DIR -Recurse -Filter "function-gpu.yaml" | ForEach-Object {
    $func_config = $_.FullName
    $func_root = Split-Path -Parent $func_config
    $func_parent_dir = Split-Path -Parent $func_root

    # Calculate relative path similar to Linux's realpath
    $func_rel_path = $func_parent_dir.Replace($SCRIPT_DIR, "").TrimStart("\")

    Write-Host "Deploying $func_rel_path function..."
    nuctl deploy --project-name cvat --path $func_root `
        --file $func_config --platform local `
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk `
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 `
        --platform-config '{\"attributes\": {\"network\": \"cvat_cvat\"}}'
}

# List deployed functions
nuctl get function --platform local
EOF
git add -f serverless/deploy_cpu.ps1 serverless/deploy_gpu.ps1
git commit -m 'Added deploy_cpu.ps1 and deploy_gpu.ps1 for Windows'

# Compatibility fix for yolov7
sed -i 's/baseImage: nvidia\/cuda:12\.6\.3-cudnn-runtime-ubuntu22\.04/baseImage: nvidia\/cuda:12\.4\.1-cudnn-runtime-ubuntu22\.04/g' serverless/onnx/WongKinYiu/yolov7/nuclio/function-gpu.yaml
git commit -m 'Downgrade cuda of yolov7 from 2.6.3 to 2.4.1 for better compatibility'

# Pop the stashed changes if necessary
 apply stash^{/local_changes}

# Resolve conflicts, test, then optionally push
git push -u origin v2.45.0-sam2 # replace with the new CVAT version you want to use
```

This approach gives a clean upgrade path while maintaining the customizations.

## Example of running SAM2 on CPU as serverless function on the same PC as CVAT:

Run on the same PC to set up the SAM2 serverless plugin for CVAT, running on the CPU (change to deploy_gpu script to run on GPU).

### For Linux PCs (Bash)

Requirements:
- Docker & Docker Compose
- NVIDIA Container Toolkit (if using deploy_gpu)

```bash
# Set variables
CVAT_ROOT_DIR=/home/jahs/GitHub/cvat # Change this to your CVAT directory
NUCLIO_BIN_DIR=/home/jahs/GitHub/bin # Change this to the directory where you want to store nuctl
USE_NUCLIO_VERSION=1.14.0 # Should match nuclio version used by CVAT

# Create directory for nuctl if it doesn't exist
mkdir -p "$NUCLIO_BIN_DIR"

# Download nuctl executable
mkdir -p "$NUCLIO_BIN_DIR"
cd "$NUCLIO_BIN_DIR"
wget "https://github.com/nuclio/nuclio/releases/download/$USE_NUCLIO_VERSION/nuctl-$USE_NUCLIO_VERSION-linux-amd64"
ln -sf "nuctl-$USE_NUCLIO_VERSION-linux-amd64" nuctl

# Navigate to CVAT root directory (assumes you are already on the correct branch, such as v2.32.0-sam2)
cd "$CVAT_ROOT_DIR"

# Add nuctl to PATH temporarily for this session
export PATH="$NUCLIO_BIN_DIR:$PATH"

# Create & start SAM2 serverless function on CPU
./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam2
```

### For Windows PCs (PowerShell)

Requirements:
- Docker & Docker Compose
  - Make sure to use the WSL2 backend for Docker Desktop
- NVIDIA Container Toolkit (if using deploy_gpu)

```powershell
# Set variables
$CVAT_ROOT_DIR = "C:\GitHub\cvat"  # Change this to your CVAT directory
$NUCLIO_BIN_DIR = "C:\GitHub\bin"  # Change this to the directory where you want to store nuctl
$USE_NUCLIO_VERSION = "1.14.0"     # Should match nuclio version used by CVAT

# Create directory for nuctl if it doesn't exist
if (-not (Test-Path -Path $NUCLIO_BIN_DIR)) {
    New-Item -ItemType Directory -Path $NUCLIO_BIN_DIR -Force
}

# Download nuctl executable
$nuctl_url = "https://github.com/nuclio/nuclio/releases/download/$USE_NUCLIO_VERSION/nuctl-$USE_NUCLIO_VERSION-windows-amd64"
$nuctl_path = Join-Path -Path $NUCLIO_BIN_DIR -ChildPath "nuctl"

Write-Host "Downloading nuctl from $nuctl_url..."
Invoke-WebRequest -Uri $nuctl_url -OutFile $nuctl_path

# Navigate to CVAT root directory (assumes you are already on the correct branch, such as v2.32.0-sam2)
Set-Location -Path $CVAT_ROOT_DIR

# Add nuctl to PATH temporarily for this session
$env:PATH = "$NUCLIO_BIN_DIR;$env:PATH"

# Create & start SAM2 serverless function on CPU
.\serverless\deploy_cpu.ps1 "$CVAT_ROOT_DIR\serverless\pytorch\facebookresearch\sam2"
```

This setup assumes that CVAT and the SAM2 serverless function are running on the same machine. Ensure that the `.env` file is configured correctly with `CVAT_NUCLIO_HOST=localhost`.

Remember to stop and restart CVAT:

```bash
cd "$CVAT_ROOT_DIR"

# Stop and remove existing containers (should use compose.yaml by default, add -f compose.yaml if not)
docker compose down

# Rebuild and restart containers (should use compose.yaml by default, add -f compose.yaml if not)
docker compose up -d --build --force-recreate --renew-anon-volumes
```

## Example of running SAM2 on GPU as serverless function on separate PC (w. IP: 172.17.155.175):

Run on separate server from CVAT server:

### For Linux PCs (Bash)

Requirements:
- Docker & Docker Compose
- NVIDIA Container Toolkit (if using deploy_gpu)

```bash
# Set variables
CVAT_ROOT_DIR=/home/kristian/GitHub/cvat # Change this to your CVAT directory
NUCLIO_BIN_DIR=/home/kristian/GitHub/bin # Change this to the directory where you want to store nuctl
USE_NUCLIO_VERSION=1.14.0 # Should match nuclio version used by CVAT
USE_NUCLIO_ADDRESS=172.17.155.175 # set to actual IP of GPU server (hostname might work, didn't in my case though)

# Create directory for nuctl if it doesn't exist
mkdir -p "$NUCLIO_BIN_DIR"

# Download nuctl executable
mkdir "$NUCLIO_BIN_DIR"
cd "$NUCLIO_BIN_DIR"
wget "https://github.com/nuclio/nuclio/releases/download/$USE_NUCLIO_VERSION/nuctl-$USE_NUCLIO_VERSION-linux-amd64"
ln -sf "nuctl-$USE_NUCLIO_VERSION-linux-amd64" nuctl

# Clone our CVAT fork and switch to branch with SAM2
git clone https://github.com/dk-teknologisk-jahs/cvat.git "$CVAT_ROOT_DIR"
cd "$CVAT_ROOT_DIR"
git switch v2.32.0-sam2

# Add nuctl to PATH temporarily for this session
export PATH="$NUCLIO_BIN_DIR:$PATH"

# The deploy_cpu/gpu scripts expect the cvat_cvat docker network to exist, so we need to create it
docker network create cvat_cvat

# Create & start SAM2 (or any other, checkout the builtin models) serverless function on GPU
PATH="$NUCLIO_BIN_DIR:$PATH" ./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam2

# Start nuclio dashboard server to monitor and provide access to the function over HTTP API
# It might be possible to get it working without this (directly access node port), but I couldn't make it work
docker run -d \
  -p 8070:8070 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --network cvat_cvat \
  --name nuclio-dashboard \
  -e NUCLIO_DASHBOARD_EXTERNAL_IP_ADDRESSES=$USE_NUCLIO_ADDRESS \
  quay.io/nuclio/dashboard:$USE_NUCLIO_VERSION-amd64
```

### For Windows PCs (PowerShell)

Requirements:
- Docker & Docker Compose
  - Make sure to use the WSL2 backend for Docker Desktop
  - If you have issues with binding to the docker daemon socket, try some of the solutions [here](https://stackoverflow.com/questions/36765138/bind-to-docker-socket-on-windows) and please create an issue with any improvement suggestions
- Ensure you have installed the NVIDIA Container Toolkit for Windows and configured Docker to use your GPU
- You might need to set the execution policy to run scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Make sure ports are open in Windows Firewall for the nuclio dashboard (8070)

```powershell
# Set variables
$CVAT_ROOT_DIR = "C:\GitHub\cvat"  # Change this to your CVAT directory
$NUCLIO_BIN_DIR = "C:\GitHub\bin"  # Change this to the directory where you want to store nuctl
$USE_NUCLIO_VERSION = "1.14.0"     # Should match nuclio version used by CVAT
$USE_NUCLIO_ADDRESS = "172.17.155.175" # set to actual IP of GPU server (hostname might work, didn't in my case though)

# Create directory for nuctl if it doesn't exist
if (-not (Test-Path -Path $NUCLIO_BIN_DIR)) {
    New-Item -ItemType Directory -Path $NUCLIO_BIN_DIR -Force
}

# Download nuctl executable
$nuctl_url = "https://github.com/nuclio/nuclio/releases/download/$USE_NUCLIO_VERSION/nuctl-$USE_NUCLIO_VERSION-windows-amd64"
$nuctl_path = Join-Path -Path $NUCLIO_BIN_DIR -ChildPath "nuctl"

Write-Host "Downloading nuctl from $nuctl_url..."
Invoke-WebRequest -Uri $nuctl_url -OutFile $nuctl_path

# Clone our CVAT fork and switch to branch with SAM2
git clone https://github.com/dk-teknologisk-jahs/cvat.git "$CVAT_ROOT_DIR"
cd "$CVAT_ROOT_DIR"
git switch v2.32.0-sam2

# Add nuctl to PATH temporarily for this session
$env:PATH = "$NUCLIO_BIN_DIR;$env:PATH"

# The deploy_cpu/gpu scripts expect the cvat_cvat docker network to exist, so we need to create it
docker network create cvat_cvat

# Create & start SAM2 (or any other, checkout the builtin models) serverless function on GPU
.\deploy_gpu.ps1 "$CVAT_ROOT_DIR\serverless\pytorch\facebookresearch\sam2"

# Start nuclio dashboard server to monitor and provide access to the function over HTTP API
# It might be possible to get it working without this (directly access node port), but I couldn't make it work
docker run -d \
  -p 8070:8070 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --network cvat_cvat \
  --name nuclio-dashboard \
  -e NUCLIO_DASHBOARD_EXTERNAL_IP_ADDRESSES=$USE_NUCLIO_ADDRESS \
  quay.io/nuclio/dashboard:$USE_NUCLIO_VERSION-amd64
```

This setup assumes that CVAT and the SAM2 serverless function are running on separate machines. Ensure that the `.env` file is configured correctly with `CVAT_NUCLIO_HOST` pointing to the IP address of the machine running the serverless functions. Be aware that you can't use both local nuclio and remote nuclio at the same time.

# Original README from here on

<p align="center">
  <img src="/site/content/en/images/cvat-readme-gif.gif" alt="CVAT Platform" width="100%" max-width="800px">
</p>
<p align="center">
  <a href="https://app.cvat.ai/">
    <img src="/site/content/en/images/cvat-readme-button-tr-bg.png" alt="Start Annotating Now">
  </a>
</p>

# Computer Vision Annotation Tool (CVAT)

[![CI][ci-img]][ci-url]
[![Gitter chat][gitter-img]][gitter-url]
[![Discord][discord-img]][discord-url]
[![Coverage Status][coverage-img]][coverage-url]
[![server pulls][docker-server-pulls-img]][docker-server-image-url]
[![ui pulls][docker-ui-pulls-img]][docker-ui-image-url]
[![DOI][doi-img]][doi-url]

CVAT is an interactive video and image annotation
tool for computer vision. It is used by tens of thousands of users and
companies around the world. Our mission is to help developers, companies, and
organizations around the world to solve real problems using the Data-centric
AI approach.

Start using CVAT online: [cvat.ai](https://cvat.ai). You can use it for free,
or [subscribe](https://www.cvat.ai/pricing/cloud) to get unlimited data,
organizations, autoannotations, and [Roboflow and HuggingFace integration](https://www.cvat.ai/post/integrating-hugging-face-and-roboflow-models).

Or set CVAT up as a self-hosted solution:
[Self-hosted Installation Guide](https://docs.cvat.ai/docs/administration/basics/installation/).
We provide [Enterprise support](https://www.cvat.ai/pricing/on-prem) for
self-hosted installations with premium features: SSO, LDAP, Roboflow and
HuggingFace integrations, and advanced analytics (coming soon). We also
do trainings and a dedicated support with 24 hour SLA.

## Quick start ⚡

- [Installation guide](https://docs.cvat.ai/docs/administration/basics/installation/)
- [Manual](https://docs.cvat.ai/docs/manual/)
- [Contributing](https://docs.cvat.ai/docs/contributing/)
- [Datumaro dataset framework](https://github.com/cvat-ai/datumaro/blob/develop/README.md)
- [Server API](#api)
- [Python SDK](#sdk)
- [Command line tool](#cli)
- [XML annotation format](https://docs.cvat.ai/docs/manual/advanced/xml_format/)
- [AWS Deployment Guide](https://docs.cvat.ai/docs/administration/basics/aws-deployment-guide/)
- [Frequently asked questions](https://docs.cvat.ai/docs/faq/)
- [Where to ask questions](#where-to-ask-questions)

## Partners ❤️

CVAT is used by teams all over the world. In the list, you can find key companies which
help us support the product or an essential part of our ecosystem. If you use us,
please drop us a line at [contact@cvat.ai](mailto:contact+github@cvat.ai).

- [Human Protocol](https://hmt.ai) uses CVAT as a way of adding annotation service to the Human Protocol.
- [FiftyOne](https://fiftyone.ai) is an open-source dataset curation and model analysis
  tool for visualizing, exploring, and improving computer vision datasets and models that are
  [tightly integrated](https://voxel51.com/docs/fiftyone/integrations/cvat.html) with CVAT
  for annotation and label refinement.

## Public datasets

[ATLANTIS](https://github.com/smhassanerfani/atlantis), an open-source dataset for semantic segmentation
of waterbody images, developed by [iWERS](http://ce.sc.edu/iwers/) group in the
Department of Civil and Environmental Engineering at the University of South Carolina is using CVAT.

For developing a semantic segmentation dataset using CVAT, see:

- [ATLANTIS published article](https://www.sciencedirect.com/science/article/pii/S1364815222000391)
- [ATLANTIS Development Kit](https://github.com/smhassanerfani/atlantis/tree/master/adk)
- [ATLANTIS annotation tutorial videos](https://www.youtube.com/playlist?list=PLIfLGY-zZChS5trt7Lc3MfNhab7OWl2BR).

## CVAT online: [cvat.ai](https://cvat.ai)

This is an online version of CVAT. It's free, efficient, and easy to use.

[cvat.ai](https://cvat.ai) runs the latest version of the tool. You can create up
to 10 tasks there and upload up to 500Mb of data to annotate. It will only be
visible to you or the people you assign to it.

For now, it does not have [analytics features](https://docs.cvat.ai/docs/administration/advanced/analytics/)
like management and monitoring the data annotation team. It also does not allow exporting images, just the annotations.

We plan to enhance [cvat.ai](https://cvat.ai) with new powerful features. Stay tuned!

## Prebuilt Docker images 🐳

Prebuilt docker images are the easiest way to start using CVAT locally. They are available on Docker Hub:

- [cvat/server](https://hub.docker.com/r/cvat/server)
- [cvat/ui](https://hub.docker.com/r/cvat/ui)

The images have been downloaded more than 1M times so far.

## Screencasts 🎦

Here are some screencasts showing how to use CVAT.

<!--lint disable maximum-line-length-->

[Computer Vision Annotation Course](https://www.youtube.com/playlist?list=PL0to7Ng4PuuYQT4eXlHb_oIlq_RPeuasN):
we introduce our course series designed to help you annotate data faster and better
using CVAT. This course is about CVAT deployment and integrations, it includes
presentations and covers the following topics:

- **Speeding up your data annotation process: introduction to CVAT and Datumaro**.
  What problems do CVAT and Datumaro solve, and how they can speed up your model
  training process. Some resources you can use to learn more about how to use them.
- **Deployment and use CVAT**. Use the app online at [app.cvat.ai](https://app.cvat.ai).
  A local deployment. A containerized local deployment with Docker Compose (for regular use),
  and a local cluster deployment with Kubernetes (for enterprise users). A 2-minute
  tour of the interface, a breakdown of CVAT’s internals, and a demonstration of how
  to deploy CVAT using Docker Compose.

[Product tour](https://www.youtube.com/playlist?list=PL0to7Ng4Puua37NJVMIShl_pzqJTigFzg): in this course, we show how to use CVAT, and help to get familiar with CVAT functionality and interfaces. This course does not cover integrations and is dedicated solely to CVAT. It covers the following topics:

- **Pipeline**. In this video, we show how to use [app.cvat.ai](https://app.cvat.ai): how to sign up, upload your data, annotate it, and download it.

<!--lint enable maximum-line-length-->

For feedback, please see [Contact us](#contact-us)

## API

- [Documentation](https://docs.cvat.ai/docs/api_sdk/api/)

## SDK

- Install with `pip install cvat-sdk`
- [PyPI package homepage](https://pypi.org/project/cvat-sdk/)
- [Documentation](https://docs.cvat.ai/docs/api_sdk/sdk/)

## CLI

- Install with `pip install cvat-cli`
- [PyPI package homepage](https://pypi.org/project/cvat-cli/)
- [Documentation](https://docs.cvat.ai/docs/api_sdk/cli/)

## Supported annotation formats

CVAT supports multiple annotation formats. You can select the format
after clicking the **Upload annotation** and **Dump annotation** buttons.
[Datumaro](https://github.com/cvat-ai/datumaro) dataset framework allows
additional dataset transformations with its command line tool and Python library.

For more information about the supported formats, see:
[Annotation Formats](https://docs.cvat.ai/docs/manual/advanced/formats/).

<!--lint disable maximum-line-length-->

| Annotation format                                                                                | Import | Export |
|--------------------------------------------------------------------------------------------------| ------ | ------ |
| [CVAT for images](https://docs.cvat.ai/docs/manual/advanced/xml_format/#annotation)              | ✔️     | ✔️     |
| [CVAT for a video](https://docs.cvat.ai/docs/manual/advanced/xml_format/#interpolation)          | ✔️     | ✔️     |
| [Datumaro](https://github.com/cvat-ai/datumaro)                                                  | ✔️     | ✔️     |
| [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)                                            | ✔️     | ✔️     |
| Segmentation masks from [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)                    | ✔️     | ✔️     |
| [YOLO](https://pjreddie.com/darknet/yolo/)                                                       | ✔️     | ✔️     |
| [MS COCO Object Detection](http://cocodataset.org/#format-data)                                  | ✔️     | ✔️     |
| [MS COCO Keypoints Detection](http://cocodataset.org/#format-data)                               | ✔️     | ✔️     |
| [MOT](https://motchallenge.net/)                                                                 | ✔️     | ✔️     |
| [MOTS PNG](https://www.vision.rwth-aachen.de/page/mots)                                          | ✔️     | ✔️     |
| [LabelMe 3.0](http://labelme.csail.mit.edu/Release3.0)                                           | ✔️     | ✔️     |
| [ImageNet](http://www.image-net.org)                                                             | ✔️     | ✔️     |
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)                             | ✔️     | ✔️     |
| [WIDER Face](http://shuoyang1213.me/WIDERFACE/)                                                  | ✔️     | ✔️     |
| [VGGFace2](https://github.com/ox-vgg/vgg_face2)                                                  | ✔️     | ✔️     |
| [Market-1501](https://www.aitribune.com/dataset/2018051063)                                      | ✔️     | ✔️     |
| [ICDAR13/15](https://rrc.cvc.uab.es/?ch=2)                                                       | ✔️     | ✔️     |
| [Open Images V6](https://storage.googleapis.com/openimages/web/index.html)                       | ✔️     | ✔️     |
| [Cityscapes](https://www.cityscapes-dataset.com/login/)                                          | ✔️     | ✔️     |
| [KITTI](http://www.cvlibs.net/datasets/kitti/)                                                   | ✔️     | ✔️     |
| [Kitti Raw Format](https://www.cvlibs.net/datasets/kitti/raw_data.php)                           | ✔️     | ✔️     |
| [LFW](http://vis-www.cs.umass.edu/lfw/)                                                          | ✔️     | ✔️     |
| [Supervisely Point Cloud Format](https://docs.supervise.ly/data-organization/00_ann_format_navi) | ✔️     | ✔️     |
| [Ultralytics YOLO Detection](https://docs.ultralytics.com/datasets/detect/)                      | ✔️     | ✔️     |
| [Ultralytics YOLO Oriented Bounding Boxes](https://docs.ultralytics.com/datasets/obb/)                     | ✔️     | ✔️     |
| [Ultralytics YOLO Segmentation](https://docs.ultralytics.com/datasets/segment/)                            | ✔️     | ✔️     |
| [Ultralytics YOLO Pose](https://docs.ultralytics.com/datasets/pose/)                                       | ✔️     | ✔️     |
| [Ultralytics YOLO Classification](https://docs.ultralytics.com/datasets/classify/)                         | ✔️     | ✔️     |

<!--lint enable maximum-line-length-->

## Deep learning serverless functions for automatic labeling

CVAT supports automatic labeling. It can speed up the annotation process
up to 10x. Here is a list of the algorithms we support, and the platforms they can be run on:

<!--lint disable maximum-line-length-->

| Name                                                                                                    | Type       | Framework  | CPU | GPU |
| ------------------------------------------------------------------------------------------------------- | ---------- | ---------- | --- | --- |
| [Segment Anything](/serverless/pytorch/facebookresearch/sam/nuclio/)                                    | interactor | PyTorch    | ✔️  | ✔️  |
| [Deep Extreme Cut](/serverless/openvino/dextr/nuclio)                                                   | interactor | OpenVINO   | ✔️  |     |
| [Faster RCNN](/serverless/openvino/omz/public/faster_rcnn_inception_resnet_v2_atrous_coco/nuclio)       | detector   | OpenVINO   | ✔️  |     |
| [Mask RCNN](/serverless/openvino/omz/public/mask_rcnn_inception_resnet_v2_atrous_coco/nuclio)           | detector   | OpenVINO   | ✔️  |     |
| [YOLO v3](/serverless/openvino/omz/public/yolo-v3-tf/nuclio)                                            | detector   | OpenVINO   | ✔️  |     |
| [YOLO v7](/serverless/onnx/WongKinYiu/yolov7/nuclio)                                                    | detector   | ONNX       | ✔️  | ✔️  |
| [Object reidentification](/serverless/openvino/omz/intel/person-reidentification-retail-0277/nuclio)    | reid       | OpenVINO   | ✔️  |     |
| [Semantic segmentation for ADAS](/serverless/openvino/omz/intel/semantic-segmentation-adas-0001/nuclio) | detector   | OpenVINO   | ✔️  |     |
| [Text detection v4](/serverless/openvino/omz/intel/text-detection-0004/nuclio)                          | detector   | OpenVINO   | ✔️  |     |
| [SiamMask](/serverless/pytorch/foolwood/siammask/nuclio)                                                | tracker    | PyTorch    | ✔️  | ✔️  |
| [TransT](/serverless/pytorch/dschoerk/transt/nuclio)                                                    | tracker    | PyTorch    | ✔️  | ✔️  |
| [f-BRS](/serverless/pytorch/saic-vul/fbrs/nuclio)                                                       | interactor | PyTorch    | ✔️  |     |
| [HRNet](/serverless/pytorch/saic-vul/hrnet/nuclio)                                                      | interactor | PyTorch    |     | ✔️  |
| [Inside-Outside Guidance](/serverless/pytorch/shiyinzhang/iog/nuclio)                                   | interactor | PyTorch    | ✔️  |     |
| [Faster RCNN](/serverless/tensorflow/faster_rcnn_inception_v2_coco/nuclio)                              | detector   | TensorFlow | ✔️  | ✔️  |
| [RetinaNet](serverless/pytorch/facebookresearch/detectron2/retinanet_r101/nuclio)                       | detector   | PyTorch    | ✔️  | ✔️  |
| [Face Detection](/serverless/openvino/omz/intel/face-detection-0205/nuclio)                             | detector   | OpenVINO   | ✔️  |     |

<!--lint enable maximum-line-length-->

## License

The code is released under the [MIT License](https://opensource.org/licenses/MIT).

The code contained within the `/serverless` directory is released under the **MIT License**.
However, it may download and utilize various assets, such as source code, architectures, and weights, among others.
These assets may be distributed under different licenses, including non-commercial licenses.
It is your responsibility to ensure compliance with the terms of these licenses before using the assets.

This software uses LGPL-licensed libraries from the [FFmpeg](https://www.ffmpeg.org) project.
The exact steps on how FFmpeg was configured and compiled can be found in the [Dockerfile](Dockerfile).

FFmpeg is an open-source framework licensed under LGPL and GPL.
See [https://www.ffmpeg.org/legal.html](https://www.ffmpeg.org/legal.html). You are solely responsible
for determining if your use of FFmpeg requires any
additional licenses. CVAT.ai Corporation is not responsible for obtaining any
such licenses, nor liable for any licensing fees due in
connection with your use of FFmpeg.

## Contact us

[Gitter](https://gitter.im/opencv-cvat/public) to ask CVAT usage-related questions.
Typically questions get answered fast by the core team or community. There you can also browse other common questions.

[Discord](https://discord.gg/S6sRHhuQ7K) is the place to also ask questions or discuss any other stuff related to CVAT.

[LinkedIn](https://www.linkedin.com/company/cvat-ai/) for the company and work-related questions.

[YouTube](https://www.youtube.com/@cvat-ai) to see screencast and tutorials about the CVAT.

[GitHub issues](https://github.com/cvat-ai/cvat/issues) for feature requests or bug reports.
If it's a bug, please add the steps to reproduce it.

[#cvat](https://stackoverflow.com/search?q=%23cvat) tag on StackOverflow is one more way to ask
questions and get our support.

[contact@cvat.ai](mailto:contact+github@cvat.ai) to reach out to us if you need commercial support.

## Links

- [Intel AI blog: New Computer Vision Tool Accelerates Annotation of Digital Images and Video](https://www.intel.ai/introducing-cvat)
- [Intel Software: Computer Vision Annotation Tool: A Universal Approach to Data Annotation](https://software.intel.com/en-us/articles/computer-vision-annotation-tool-a-universal-approach-to-data-annotation)
- [VentureBeat: Intel open-sources CVAT, a toolkit for data labeling](https://venturebeat.com/2019/03/05/intel-open-sources-cvat-a-toolkit-for-data-labeling/)
- [How to Use CVAT (Roboflow guide)](https://blog.roboflow.com/cvat/)
- [How to auto-label data in CVAT with one of 50,000+ models on Roboflow Universe](https://blog.roboflow.com/how-to-use-roboflow-models-in-cvat/)

  <!-- Badges -->

[docker-server-pulls-img]: https://img.shields.io/docker/pulls/cvat/server.svg?style=flat-square&label=server%20pulls
[docker-server-image-url]: https://hub.docker.com/r/cvat/server
[docker-ui-pulls-img]: https://img.shields.io/docker/pulls/cvat/ui.svg?style=flat-square&label=UI%20pulls
[docker-ui-image-url]: https://hub.docker.com/r/cvat/ui
[ci-img]: https://github.com/cvat-ai/cvat/actions/workflows/main.yml/badge.svg?branch=develop
[ci-url]: https://github.com/cvat-ai/cvat/actions
[gitter-img]: https://img.shields.io/gitter/room/opencv-cvat/public?style=flat
[gitter-url]: https://gitter.im/opencv-cvat/public
[coverage-img]: https://codecov.io/github/cvat-ai/cvat/branch/develop/graph/badge.svg
[coverage-url]: https://codecov.io/github/cvat-ai/cvat
[doi-img]: https://zenodo.org/badge/139156354.svg
[doi-url]: https://zenodo.org/badge/latestdoi/139156354
[discord-img]: https://img.shields.io/discord/1000789942802337834?label=discord
[discord-url]: https://discord.gg/fNR3eXfk6C
