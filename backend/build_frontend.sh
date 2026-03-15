#!/bin/bash
# Build script to compile React frontend and move to FastAPI static directory

echo "🔨 Building React frontend..."

# Navigate to frontend directory
cd ../StrokeMitra

# Install dependencies
npm install

# Build for production
npm run build

# Create static directory in backend if it doesn't exist
mkdir -p ../slurry_speech/static

# Copy built files to backend's static directory
echo "📦 Copying build files to backend..."
rm -rf ../slurry_speech/static/*
cp -r dist/* ../slurry_speech/static/

echo "✅ Frontend built and copied to backend/static/"
