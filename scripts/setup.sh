#!/bin/bash

# ToolHop Plan Generator - Quick Setup Script
# This script helps you get started quickly

set -e  # Exit on error

echo "=================================================="
echo "ToolHop Plan Generator - Quick Setup"
echo "=================================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if API key is set
echo ""
echo "Checking for OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    read -p "Enter your OpenAI API key now (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
        echo "✓ API key set for this session"
    fi
else
    echo "✓ API key found"
fi

# Check if ToolHop data exists
echo ""
echo "Checking for ToolHop dataset..."
if [ ! -f "toolhop.json" ]; then
    echo "⚠️  WARNING: toolhop.json not found in current directory"
    echo "Please place your ToolHop dataset file here or specify the path when running scripts"
else
    echo "✓ Found toolhop.json"
fi

# Create output directory
echo ""
echo "Creating output directory..."
mkdir -p outputs
echo "✓ Created ./outputs/"

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. RECOMMENDED: Run tests first (no API calls, ~5 minutes)"
echo "   python test_plan_generator.py --toolhop-path toolhop.json --api-key \$OPENAI_API_KEY --test all"
echo ""
echo "2. Generate a small test dataset (10 queries, ~$5, ~30 minutes)"
echo "   python toolhop_plan_generator.py --toolhop-path toolhop.json --output-path outputs/test.json --api-key \$OPENAI_API_KEY --max-queries 10 --n-candidates 10"
echo ""
echo "3. Validate the test output"
echo "   python test_plan_generator.py --toolhop-path toolhop.json --api-key \$OPENAI_API_KEY --validate-dataset outputs/test.json"
echo ""
echo "4. If everything looks good, generate the full dataset (all queries, ~$200-300, ~3-5 hours)"
echo "   python toolhop_plan_generator.py --toolhop-path toolhop.json --output-path outputs/full.json --api-key \$OPENAI_API_KEY --n-candidates 10"
echo ""
echo "See README.md and README_USAGE.md for detailed documentation."
echo ""
