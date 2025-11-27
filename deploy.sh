#!/bin/bash

# Deployment script for Intelligent Content Organizer to Blaxel
# Usage: ./deploy.sh [environment]
# environment: dev, staging, production (default: production)

set -e  # Exit on error

ENVIRONMENT=${1:-production}
APP_NAME="AI-Digital-Library-Assistant"

echo "🚀 Deploying $APP_NAME to Blaxel ($ENVIRONMENT environment)..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Blaxel CLI is installed
if ! command -v bl &> /dev/null; then
    print_error "Blaxel CLI not found. Installing..."
    pip install blaxel-cli
    
    if ! command -v bl &> /dev/null; then
        print_error "Failed to install Blaxel CLI. Please install manually:"
        echo "  pip install blaxel-cli"
        exit 1
    fi
fi

print_status "Blaxel CLI found"

# Check if logged in
if ! bl auth whoami &> /dev/null; then
    print_warning "Not logged in to Blaxel. Starting login..."
    bl auth login
fi

print_status "Authenticated with Blaxel"

# Validate configuration
echo ""
echo "📋 Validating blaxel.yaml..."
if bl validate blaxel.yaml; then
    print_status "Configuration valid"
else
    print_error "Configuration validation failed. Please check blaxel.yaml"
    exit 1
fi

# Check environment variables
echo ""
echo "🔐 Checking environment variables..."
required_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "ELEVENLABS_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    print_warning "Missing environment variables: ${missing_vars[*]}"
    echo "Make sure to set these in your Blaxel dashboard or .env file"
else
    print_status "All required environment variables found"
fi

# Build deployment package
echo ""
echo "📦 Building deployment package..."
if [ -f "requirements.txt" ]; then
    print_status "Found requirements.txt"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Deploy to Blaxel
echo ""
echo "🚢 Deploying to Blaxel ($ENVIRONMENT)..."
bl deploy \
    --config blaxel.yaml \
    --env $ENVIRONMENT \
    --name $APP_NAME \
    --yes

if [ $? -eq 0 ]; then
    print_status "Deployment successful!"
else
    print_error "Deployment failed"
    exit 1
fi

# Get deployment info
echo ""
echo "📊 Deployment Information:"
bl status $APP_NAME --env $ENVIRONMENT

# Show logs
echo ""
echo "Would you like to view deployment logs? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    bl logs $APP_NAME --env $ENVIRONMENT --follow
fi

echo ""
print_status "Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Configure Claude Desktop with MCP server URL"
echo "  2. Test MCP tools via Claude"
echo "  3. Monitor with: bl logs $APP_NAME --env $ENVIRONMENT"
echo "  4. Manage at: bl dashboard"
echo ""
