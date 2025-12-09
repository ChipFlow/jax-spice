#!/bin/bash
# Setup script for GCP TPU CI infrastructure
# This script is fully idempotent - safe to run multiple times
#
# Usage: ./setup_gcp_tpu_ci.sh
#
# Features:
# - Enables TPU API
# - Creates/updates service account with TPU permissions
# - Stores service account key in GCP Secret Manager
# - Syncs secret to GitHub Actions
#
# Prerequisites:
# - gcloud CLI installed and authenticated
# - gh CLI installed and authenticated (for GitHub secret sync)
# - Billing enabled on the GCP project
# - TPU quota available in the zone (request at https://cloud.google.com/tpu/docs/quota)
#
# TPU Quota:
# - You need quota for the TPU type you want to use
# - For v5e: request "TPU v5 Lite PodSlice chips" quota
# - For Spot VMs: request preemptible quota separately
# - Quota request: https://console.cloud.google.com/iam-admin/quotas

set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT:-jax-spice-cuda-test}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"  # Must have TPU v5e availability
SA_NAME="github-gpu-ci"  # Reuse existing service account
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SECRET_NAME="github-gpu-ci-key"
GITHUB_REPO="${GITHUB_REPO:-ChipFlow/jax-spice}"

# TPU Configuration
TPU_TYPE="${TPU_TYPE:-v5litepod-8}"  # Smallest v5e configuration
TPU_RUNTIME="${TPU_RUNTIME:-v2-alpha-tpuv5-lite}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  JAX-SPICE TPU CI Setup (Idempotent)"
echo "=========================================="
echo ""
echo "Project:     ${PROJECT_ID}"
echo "Zone:        ${ZONE}"
echo "TPU Type:    ${TPU_TYPE}"
echo "Runtime:     ${TPU_RUNTIME}"
echo "GitHub Repo: ${GITHUB_REPO}"
echo ""

# Set project
gcloud config set project "${PROJECT_ID}" --quiet

# =============================================================================
# Step 1: Enable required APIs (idempotent)
# =============================================================================
log_info "Enabling required APIs..."
gcloud services enable tpu.googleapis.com --quiet
gcloud services enable compute.googleapis.com --quiet
gcloud services enable secretmanager.googleapis.com --quiet
gcloud services enable iam.googleapis.com --quiet
log_info "APIs enabled"

# =============================================================================
# Step 2: Check/create service account (idempotent)
# =============================================================================
log_info "Setting up service account..."
if gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    log_info "Service account already exists: ${SA_EMAIL}"
else
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="GitHub GPU/TPU CI Runner" \
        --quiet
    log_info "Created service account: ${SA_EMAIL}"
fi

# =============================================================================
# Step 3: Grant TPU permissions (idempotent)
# =============================================================================
log_info "Configuring IAM permissions for TPU..."

# Define required roles for TPU access
ROLES=(
    "roles/tpu.admin"                  # Create/delete/manage TPU VMs
    "roles/compute.networkUser"        # Use VPC networks for TPU
    "roles/iam.serviceAccountUser"     # Use service account on TPU VM
    "roles/logging.viewer"             # View logs
    "roles/storage.objectViewer"       # Read from GCS (for JAX wheels)
)

for ROLE in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet \
        --condition=None 2>/dev/null || true
done
log_info "IAM permissions configured"

# =============================================================================
# Step 4: Check TPU quota
# =============================================================================
log_info "Checking TPU quota..."

# Extract chip count from TPU type (e.g., v5litepod-8 -> 8)
CHIP_COUNT=$(echo "${TPU_TYPE}" | grep -oE '[0-9]+$')

echo ""
echo "Required quota for ${TPU_TYPE}:"
echo "  - TPU v5 Lite PodSlice chips: ${CHIP_COUNT} (on-demand)"
echo "  - Preemptible TPU v5 Lite PodSlice chips: ${CHIP_COUNT} (for Spot VMs)"
echo ""
echo "Check your quota at:"
echo "  https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
echo ""
echo "Filter by: 'tpu' and region '${REGION}'"
echo ""

# =============================================================================
# Step 5: Verify TPU availability in zone
# =============================================================================
log_info "Checking TPU availability in ${ZONE}..."

# List available accelerator types
AVAILABLE_TYPES=$(gcloud compute tpus accelerator-types list \
    --zone="${ZONE}" \
    --format="value(type)" 2>/dev/null | grep -E "^v5litepod" || echo "")

if [ -z "${AVAILABLE_TYPES}" ]; then
    log_warn "No v5e TPUs found in ${ZONE}. Available zones for v5e:"
    echo "  - us-central1-a"
    echo "  - us-south1-a"
    echo "  - us-west1-c"
    echo "  - us-west4-a"
    echo "  - europe-west4-b"
    echo ""
    echo "Update GCP_ZONE environment variable and re-run."
else
    log_info "Available TPU types in ${ZONE}:"
    echo "${AVAILABLE_TYPES}" | sed 's/^/  - /'
fi

# =============================================================================
# Step 6: Create/update secret in Secret Manager (idempotent)
# =============================================================================
log_info "Setting up Secret Manager..."

# Check if secret exists
if gcloud secrets describe "${SECRET_NAME}" &>/dev/null; then
    log_info "Secret already exists: ${SECRET_NAME}"

    # Check if we need to create a new key version
    LATEST_VERSION=$(gcloud secrets versions list "${SECRET_NAME}" \
        --filter="state=ENABLED" \
        --sort-by="~createTime" \
        --limit=1 \
        --format="value(name)" 2>/dev/null || echo "")

    if [ -n "${LATEST_VERSION}" ]; then
        log_info "Using existing secret version"
        NEED_NEW_KEY=false
    else
        log_warn "No enabled secret versions found, creating new key"
        NEED_NEW_KEY=true
    fi
else
    log_info "Creating new secret: ${SECRET_NAME}"
    gcloud secrets create "${SECRET_NAME}" \
        --replication-policy="automatic" \
        --quiet
    NEED_NEW_KEY=true
fi

# Create new key and add to secret if needed
if [ "${NEED_NEW_KEY:-false}" = true ]; then
    log_info "Generating new service account key..."

    # Create temporary key file
    KEY_FILE=$(mktemp)
    trap "rm -f ${KEY_FILE}" EXIT

    gcloud iam service-accounts keys create "${KEY_FILE}" \
        --iam-account="${SA_EMAIL}" \
        --quiet

    # Add new version to secret
    gcloud secrets versions add "${SECRET_NAME}" \
        --data-file="${KEY_FILE}" \
        --quiet

    log_info "Service account key stored in Secret Manager"

    # Clean up old keys (keep only the 2 most recent)
    log_info "Cleaning up old service account keys..."
    OLD_KEYS=$(gcloud iam service-accounts keys list \
        --iam-account="${SA_EMAIL}" \
        --format="value(name)" \
        --filter="keyType=USER_MANAGED" \
        --sort-by="~validAfterTime" 2>/dev/null | tail -n +3)

    for KEY_ID in ${OLD_KEYS}; do
        gcloud iam service-accounts keys delete "${KEY_ID}" \
            --iam-account="${SA_EMAIL}" \
            --quiet 2>/dev/null || true
    done
fi

# =============================================================================
# Step 7: Sync secret to GitHub (idempotent)
# =============================================================================
log_info "Syncing secret to GitHub..."

if command -v gh &>/dev/null; then
    if gh auth status &>/dev/null; then
        SECRET_VALUE=$(gcloud secrets versions access latest --secret="${SECRET_NAME}" 2>/dev/null)

        if [ -n "${SECRET_VALUE}" ]; then
            echo "${SECRET_VALUE}" | gh secret set GCP_SERVICE_ACCOUNT_KEY \
                --repo="${GITHUB_REPO}" 2>/dev/null && \
                log_info "GitHub secret 'GCP_SERVICE_ACCOUNT_KEY' updated" || \
                log_warn "Failed to update GitHub secret (check gh permissions)"
        else
            log_error "Could not retrieve secret from Secret Manager"
        fi
    else
        log_warn "gh CLI not authenticated. Run 'gh auth login' to sync secrets"
    fi
else
    log_warn "gh CLI not installed. Install it to auto-sync GitHub secrets"
fi

# =============================================================================
# Step 8: Test TPU creation (optional, commented out)
# =============================================================================
# Uncomment to test TPU VM creation:
#
# log_info "Testing TPU VM creation..."
# gcloud compute tpus tpu-vm create test-tpu-vm \
#     --zone="${ZONE}" \
#     --accelerator-type="${TPU_TYPE}" \
#     --version="${TPU_RUNTIME}" \
#     --spot \
#     --quiet
#
# log_info "TPU VM created successfully, deleting..."
# gcloud compute tpus tpu-vm delete test-tpu-vm \
#     --zone="${ZONE}" \
#     --quiet

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "  TPU CI Setup Complete!"
echo "=========================================="
echo ""
echo "Resources configured:"
echo "  - Service Account: ${SA_EMAIL}"
echo "  - Secret (GCP):    ${SECRET_NAME}"
echo "  - GitHub Secret:   GCP_SERVICE_ACCOUNT_KEY (shared with GPU CI)"
echo ""
echo "TPU Configuration:"
echo "  - Type:    ${TPU_TYPE} (8 chips, 128 GB HBM2)"
echo "  - Zone:    ${ZONE}"
echo "  - Runtime: ${TPU_RUNTIME}"
echo ""
echo "Estimated costs:"
echo "  - On-demand: ~\$9.60/hour (8 × \$1.20/chip)"
echo "  - Spot:      ~\$1-2/hour (up to 91% discount)"
echo ""
echo "Next steps:"
echo "  1. Request TPU quota if needed (link above)"
echo "  2. Push the test-tpu.yml workflow"
echo "  3. Trigger workflow manually or via PR"
echo ""
