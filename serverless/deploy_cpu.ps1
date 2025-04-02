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
