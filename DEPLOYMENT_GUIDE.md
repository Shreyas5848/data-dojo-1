# DataDojo Deployment Guide 🚀

## Quick Start Options

### 🖥️ **Local CLI Launch (Recommended for Development)**
```bash
# Navigate to project directory
cd data-dojo-1

# Install dependencies
pip install -r requirements.txt

# Launch web dashboard via CLI
python -m src.datadojo web

# Or specify custom port
python -m src.datadojo web --port 8503

# Launch without opening browser
python -m src.datadojo web --no-browser

# Check if dashboard is running
python -m src.datadojo web --status
```

### 🌐 **Direct Streamlit Launch**
```bash
# Traditional Streamlit launch
streamlit run app.py

# With custom port
streamlit run app.py --server.port 8502
```

## Cloud Deployment Options

### ☁️ **Streamlit Community Cloud (Free)**

**Prerequisites:**
- GitHub repository (public)
- Streamlit Community Cloud account

**Steps:**
1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add DataDojo web dashboard"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configuration Files (Already included):**
   - `.streamlit/config.toml` - Theme and settings
   - `requirements.txt` - Dependencies
   - `demo_datasets/` - Sample data for demo

**Deployment URL:** `https://your-repo-name.streamlit.app`

### 🐳 **Docker Deployment**

**Create Dockerfile:**
```dockerfile
# Use official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

**Docker Commands:**
```bash
# Build image
docker build -t datadojo-web .

# Run container
docker run -p 8501:8501 datadojo-web

# Run with volume for data persistence
docker run -p 8501:8501 -v $(pwd)/data:/app/data datadojo-web
```

### 🌊 **Heroku Deployment**

**Create Procfile:**
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

**Deploy Commands:**
```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create datadojo-web-app

# Set buildpacks
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

## Environment Configuration

### **Environment Variables**
```bash
# Optional configuration
export DATADOJO_ENV=production
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### **Configuration Files**

**`.streamlit/config.toml`:**
```toml
[theme]
primaryColor = "#FF9900"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
textColor = "#2E3440"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

**`requirements.txt`:**
```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
# ... other dependencies
```

## Demo Data Setup

### **Included Demo Datasets**
- `demo_datasets/sample_patients.csv` (35.4 KB)
- `demo_datasets/sample_lab_results.csv` (37.5 KB)
- `demo_datasets/sample_customers.csv` (20.1 KB)
- `demo_datasets/sample_transactions.csv` (40.6 KB)
- `demo_datasets/sample_bank_transactions.csv` (32.3 KB)
- `demo_datasets/sample_credit_applications.csv` (19.4 KB)

### **Generate Fresh Demo Data**
```bash
# Generate new demo datasets
python -c "
from src.datadojo.utils.synthetic_data_generator import SyntheticDataGenerator
generator = SyntheticDataGenerator()
# ... generation code
"
```

## Troubleshooting

### **Common Issues**

**Port Already in Use:**
```bash
# Find process using port 8501
netstat -ano | findstr :8501  # Windows
lsof -ti:8501                 # macOS/Linux

# Kill process
taskkill /PID <PID> /F        # Windows
kill -9 <PID>                # macOS/Linux
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

**Streamlit Config Issues:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Check config
streamlit config show
```

### **Performance Optimization**

**For Large Datasets:**
- Enable data sampling in configuration
- Increase memory limits for cloud deployment
- Use data caching strategies

**For Multiple Users:**
- Consider load balancing
- Enable session state optimization
- Monitor resource usage

## Production Checklist

### **Before Deployment:**
- [ ] Test all features locally
- [ ] Verify demo datasets load correctly
- [ ] Check error handling for missing files
- [ ] Validate visualizations render properly
- [ ] Test on different screen sizes

### **Security Considerations:**
- [ ] Remove debug settings in production
- [ ] Validate file upload restrictions
- [ ] Check data privacy compliance
- [ ] Review error message exposure

### **Monitoring:**
- [ ] Set up logging
- [ ] Monitor resource usage
- [ ] Track user interactions
- [ ] Set up alerts for errors

## Support & Resources

### **DataDojo Specific:**
- CLI Help: `python -m src.datadojo web --help`
- Generate Data: `python -m src.datadojo generate-data`
- Profile Data: `python -m src.datadojo profile-data`

### **Streamlit Resources:**
- [Streamlit Documentation](https://docs.streamlit.io)
- [Community Cloud Guide](https://docs.streamlit.io/streamlit-community-cloud)
- [Deployment Best Practices](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)

### **Getting Help:**
- Check application logs for detailed error messages
- Use `--debug` flag for verbose output
- Test with sample datasets first
- Verify all dependencies are installed

---

**🎉 Your DataDojo web dashboard is now ready for deployment!**

Choose the deployment method that best fits your needs:
- **Local Development**: CLI launch with `python -m src.datadojo web`
- **Public Demo**: Streamlit Community Cloud
- **Production**: Docker or Heroku deployment