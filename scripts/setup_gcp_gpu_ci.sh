#!/bin/bash
# Setup script for GCP GPU CI infrastructure
# This script creates the service account and VM needed for GPU CI testing
#
# Prerequisites:
# - gcloud CLI installed and authenticated
# - Billing enabled on the GCP project
# - GPU quota available in the region

set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT:-jax-spice}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${GCP_VM_NAME:-jax-spice-cuda}"
SA_NAME="github-gpu-ci"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Machine configuration
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "=== JAX-SPICE GPU CI Setup ==="
echo "Project: ${PROJECT_ID}"
echo "Zone: ${ZONE}"
echo "VM Name: ${VM_NAME}"
echo ""

# Set project
gcloud config set project "${PROJECT_ID}"

# 1. Enable required APIs
echo "=== Enabling APIs ==="
gcloud services enable compute.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable iam.googleapis.com

# 2. Create service account for GitHub Actions
echo ""
echo "=== Creating Service Account ==="
if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="GitHub GPU CI Runner"
    echo "Created service account: ${SA_EMAIL}"
else
    echo "Service account already exists: ${SA_EMAIL}"
fi

# 3. Grant permissions to service account
echo ""
echo "=== Granting Permissions ==="
# Compute instance admin for starting/stopping VMs
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/compute.instanceAdmin.v1" \
    --quiet

# Service account user to run commands as the VM's service account
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/iam.serviceAccountUser" \
    --quiet

# OS Login for SSH access
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/compute.osLogin" \
    --quiet

echo "Permissions granted"

# 4. Create key for GitHub Actions secret
echo ""
echo "=== Creating Service Account Key ==="
KEY_FILE="/tmp/${SA_NAME}-key.json"
gcloud iam service-accounts keys create "${KEY_FILE}" \
    --iam-account="${SA_EMAIL}"

echo ""
echo "Service account key created at: ${KEY_FILE}"
echo ""
echo "IMPORTANT: Add this key as a GitHub secret named 'GCP_SERVICE_ACCOUNT_KEY'"
echo "You can copy the content with:"
echo "  cat ${KEY_FILE}"
echo ""
echo "Then delete the local key file:"
echo "  rm ${KEY_FILE}"
echo ""

# 5. Create GPU VM (if it doesn't exist)
echo "=== Creating GPU VM ==="
if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" &>/dev/null; then
    echo "VM already exists: ${VM_NAME}"
else
    gcloud compute instances create "${VM_NAME}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type="pd-ssd" \
        --maintenance-policy="TERMINATE" \
        --no-restart-on-failure \
        --metadata="install-nvidia-driver=True" \
        --scopes="cloud-platform"

    echo "Created VM: ${VM_NAME}"

    # Wait for VM to be ready
    echo "Waiting for VM to initialize..."
    sleep 60
fi

# 6. Setup Python environment on VM
echo ""
echo "=== Setting up Python environment on VM ==="
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="
    # Install Python 3.10 if not present
    if ! command -v python3.10 &>/dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv
    fi

    # Create working directory
    mkdir -p ~/jax-spice-ci

    # Verify GPU is accessible
    nvidia-smi

    echo ''
    echo 'GPU VM setup complete!'
"

# 7. Stop VM to save costs
echo ""
echo "=== Stopping VM to save costs ==="
gcloud compute instances stop "${VM_NAME}" --zone="${ZONE}"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Add the service account key to GitHub secrets as 'GCP_SERVICE_ACCOUNT_KEY'"
echo "2. Push the workflow file to trigger GPU tests"
echo "3. The workflow will start/stop the VM as needed"
echo ""
echo "To manually test, start the VM with:"
echo "  gcloud compute instances start ${VM_NAME} --zone=${ZONE}"
echo ""
echo "To SSH into the VM:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
