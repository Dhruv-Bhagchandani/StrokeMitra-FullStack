#!/bin/bash
# Kaggle Setup Helper Script

set -e

echo "════════════════════════════════════════════════════════════════════════════"
echo "  Kaggle API Setup for Dataset Download"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Install Kaggle API
echo "Step 1: Installing Kaggle API..."
pip3 install kaggle --upgrade --quiet
echo "✓ Kaggle API installed"
echo ""

# Step 2: Check for credentials
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✓ Kaggle credentials already exist at ~/.kaggle/kaggle.json"
else
    echo "⚠️  Kaggle credentials not found!"
    echo ""
    echo "To download datasets from Kaggle, you need API credentials."
    echo ""
    echo "📝 How to get your Kaggle API token:"
    echo ""
    echo "1. Go to: https://www.kaggle.com/account"
    echo "   (You may need to sign in or create a free account)"
    echo ""
    echo "2. Scroll down to 'API' section"
    echo ""
    echo "3. Click 'Create New Token'"
    echo "   This will download 'kaggle.json' to your Downloads folder"
    echo ""
    echo "4. Once you have the file, come back here!"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    read -p "Do you have the kaggle.json file? (y/n): " has_file

    if [ "$has_file" = "y" ] || [ "$has_file" = "Y" ]; then
        echo ""
        echo "Great! Where is your kaggle.json file located?"
        echo "(Common locations: ~/Downloads/kaggle.json or ~/Desktop/kaggle.json)"
        echo ""
        read -p "Enter full path to kaggle.json: " kaggle_path

        # Expand ~ to home directory
        kaggle_path="${kaggle_path/#\~/$HOME}"

        if [ -f "$kaggle_path" ]; then
            # Create .kaggle directory
            mkdir -p ~/.kaggle

            # Copy file
            cp "$kaggle_path" ~/.kaggle/kaggle.json

            # Set permissions
            chmod 600 ~/.kaggle/kaggle.json

            echo ""
            echo "✓ Kaggle credentials installed successfully!"
            echo "✓ Permissions set (600)"
        else
            echo ""
            echo "❌ File not found at: $kaggle_path"
            echo "Please check the path and try again."
            exit 1
        fi
    else
        echo ""
        echo "Please complete the steps above and run this script again."
        echo ""
        echo "Quick command after downloading:"
        echo "  mkdir -p ~/.kaggle"
        echo "  cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
        echo "  chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  Testing Kaggle API Connection"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Test connection
echo "Testing connection to Kaggle..."
kaggle datasets list --max-size 1 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Connection successful!"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  Ready to Download Dataset"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Dataset: dysarthria-and-nondysarthria-speech-dataset"
    echo "Source: Kaggle (poojag718)"
    echo ""
    read -p "Download dataset now? (y/n): " download_now

    if [ "$download_now" = "y" ] || [ "$download_now" = "Y" ]; then
        echo ""
        echo "Starting download..."
        python3 scripts/download_dataset.py
    else
        echo ""
        echo "You can download later by running:"
        echo "  python3 scripts/download_dataset.py"
    fi
else
    echo "❌ Connection failed. Please check your credentials."
    echo ""
    echo "Verify your kaggle.json file at: ~/.kaggle/kaggle.json"
    echo "It should look like:"
    echo '  {"username":"your_username","key":"your_api_key"}'
    exit 1
fi

echo ""
echo "✓ Setup complete!"
echo ""
