# AAC Communication System 🗣️

An advanced Augmentative and Alternative Communication (AAC) application built with Python and Kivy, featuring AI-powered image generation, intelligent text processing, speech synthesis, and context-aware recommendations for users with communication difficulties.

## 🌟 Features

### Core Communication Tools
- **Text-to-Speech (TTS)** - Convert text to natural speech using gTTS
- **Visual Communication** - Picture-based communication with ARASAAC pictograms
- **Voice Input** - Speech recognition for hands-free text input
- **Touch Interface** - Intuitive touch-based interaction for all users
- **Multi-modal Output** - Combined visual, audio, and text feedback

### AI-Powered Intelligence
- **DALL-E Image Generation** - Generate custom images for words/sentences not in database
- **Smart Text Matching** - Fuzzy matching and similarity scoring using TF-IDF
- **AI Text Normalization** - Automatic correction and enhancement of input text
- **Context-Aware Suggestions** - Time and location-based intelligent recommendations
- **Multi-word Processing** - Advanced sentence parsing and semantic matching

### Intelligent Database System
- **SQLite Integration** - Local database for storing user patterns and preferences
- **TF-IDF Analysis** - Sophisticated ranking of sentence suggestions
- **Time-Phase Tracking** - Context-aware recommendations based on time of day
- **Location-Based Learning** - Adaptive suggestions based on user location context
- **Usage Analytics** - Track and analyze communication patterns for better suggestions

### Data Management & Sync
- **Google Drive Integration** - Automatic image backup and synchronization
- **CSV Database** - Structured storage of sentences, labels, and categories
- **Metadata Management** - Comprehensive tracking of image sources and pictogram data
- **Offline Capability** - Full functionality with local caching when offline
- **Cloud Backup** - Secure backup of generated content and user preferences

### Advanced UI/UX
- **Responsive Design** - Adapts to different screen sizes and orientations
- **Accessibility Features** - High contrast, large buttons, clear typography
- **Category Organization** - Intuitive grouping of communication items
- **Visual Feedback** - Clear indicators for user interactions and system status
- **Customizable Interface** - Adaptable to different user needs and preferences

## 🚀 Quick Start
### Prerequisites
- **Python 3.8+** - Required for all features
- **Virtual Environment** - Recommended for clean installation
- **Google Cloud Account** - For Drive API integration
- **OpenAI Account** - For DALL-E image generation
- **Microphone** - For speech recognition features (optional)
- **Internet Connection** - For AI features and initial setup

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/KSKAUSHIKRAM/CPR
cd AAC-25.11.25
```
2. **Create and activate virtual environment**
```bash
# Create virtual environment
python -m venv myvirtual

# Activate virtual environment
# Windows:
myvirtual\Scripts\activate
# macOS/Linux:
source myvirtual/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models**
```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"
```

### Configuration

#### 1. OpenAI API Setup
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your OpenAI API key
# Use any text editor (notepad, nano, vim, etc.)
```

**Add to .env file:**
```env
OPENAI_API_KEY=sk-proj-your_actual_api_key_here
```

#### 2. Google Drive API Setup
1. **Go to [Google Cloud Console](https://console.cloud.google.com/)**
2. **Create a new project** or select existing one
3. **Enable Google Drive API** for your project
4. **Create OAuth 2.0 credentials** (Desktop Application)
5. **Download the JSON credentials** and save as `View/client_secrets.json`

```bash
# Copy template and fill in your credentials
cp client_secrets_template.json View/client_secrets.json
# Edit View/client_secrets.json with your Google Cloud credentials
```

#### 3. Database Setup
The SQLite database is created automatically on first run. No additional configuration needed.

#### 4. Test Configuration
```bash
# Test OpenAI API key
python test_env.py

# Expected output:
# ✅ Environment variable loaded successfully!
# API Key starts with: sk-proj-...
```

### Running the Application

```bash
# Ensure virtual environment is activated
myvirtual\Scripts\activate  # Windows
source myvirtual/bin/activate  # macOS/Linux

# Run the application
python main.py
```

## 📁 Project Structure

```
AAC-25.11.25/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in repo)
├── .env.example               # Environment template
├── test_env.py                # Configuration test script
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
│
├── View/                      # User Interface Layer
│   ├── AACScreen.py          # Main screen and UI logic
│   ├── client_secrets.json   # Google Drive API credentials (not in repo)
│   ├── mycreds.json         # Saved Google Drive tokens (not in repo)
│   └── icons/               # UI icons and images
│
├── Control/                   # Application Controller
│   └── Controller.py         # Main application controller
│
└── Model/                     # Data Layer
    ├── Database.py           # SQLite database operations
    ├── dataset.csv           # Communication dataset
    ├── metadata_drive.json   # Image metadata and URLs
    └── Database/             # SQLite database storage
        └── aac.db           # SQLite database file (auto-created)
```

## 🎯 Usage Guide

### Basic Communication
1. **Launch the application** - Run `python main.py`
2. **Type or speak** your message in the input field
3. **Press GO** to process the input and generate suggestions
4. **View results** in the pictogram grid with visual representations
5. **Tap images** to hear speech output via text-to-speech
6. **Select from suggestions** in the bottom panel for quick access

### Advanced Features

#### Multi-word Processing
- **Type complex sentences** like "I want to drink water"
- **AI matching** finds the best related content from your database
- **Contextual suggestions** based on time of day and usage patterns
- **Fallback generation** creates new images if no match is found

#### Voice Input
- **Click microphone** button for speech recognition
- **Speak naturally** - the system processes and corrects input
- **Hands-free operation** for users with limited mobility

#### Smart Recommendations
- **Time-aware suggestions** - Different content for morning, afternoon, evening
- **Location context** - Suggestions adapt to your current environment
- **Usage patterns** - Learns from your communication habits
- **TF-IDF ranking** - Intelligent scoring of relevance

### Content Management

#### Adding New Content
1. **Add to Category**: Use the "+" button to add items to existing categories
2. **Create Category**: Generate new categories with custom DALL-E images
3. **Custom Images**: Generate unique images for missing pictograms
4. **Bulk Import**: Add multiple items via CSV file editing

#### Database Management
- **Automatic learning** - System learns from your usage patterns
- **Context tracking** - Records time, location, and frequency of use
- **Intelligent suggestions** - Improves recommendations over time
- **Data export** - Access your usage data for analysis

## 🔧 Configuration & Customization

### Environment Variables (.env)
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional Configuration
DEFAULT_VOICE_LANG=en
CACHE_EXPIRY_DAYS=30
MAX_SUGGESTIONS=10
TTS_VOICE_SPEED=1.0
```

### Google Drive Settings
- **Target Folder ID**: `1X1ya6OLQA9SBIcicaaceBAukaV94RpOB`
- **OAuth Scopes**: Google Drive read/write access
- **Automatic Sync**: Images uploaded and synced automatically
- **Offline Mode**: Local cache available when offline

### Database Configuration
- **Database Location**: `Model/Database/aac.db`
- **Auto-creation**: Database and tables created on first run
- **Time Phases**: Morning (6-10), Midday (10-14), Afternoon (14-18), Evening (18-21), Night (21-6)
- **TF-IDF Settings**: Alpha parameter (0.6) balances frequency vs. uniqueness

## 📊 Data Formats & Schema

### Communication Dataset (dataset.csv)
```csv
yes_sentence,label,category
"I want water","water","drinks"
"Good morning everyone","greeting","social"
"I need help please","help","requests"
"Time to eat lunch","lunch","meals"
"Let's go outside","outside","activities"
```

### Database Schema (SQLite)
```sql
CREATE TABLE Sentences (
    sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    time_phase TEXT,           -- Morning, Midday, Afternoon, Evening, Night
    location_tag TEXT,         -- User-defined location context
    last_used_at TIME DEFAULT (TIME('now', 'localtime')),
    day TEXT                   -- Day of week for pattern analysis
);
```

### Image Metadata (metadata_drive.json)
```json
{
  "water": {
    "filename": "water_icon.png",
    "pic_id": 123,
    "url": "https://drive.google.com/uc?export=view&id=1D9fy4yYtl7L..."
  },
  "greeting": {
    "filename": "dalle_greeting.png", 
    "pic_id": 124,
    "url": "https://drive.google.com/uc?export=view&id=164pqcHvCp..."
  }
}
```

## 📦 Dependencies

### Core Python Libraries (Built-in)
- `sqlite3` - Database operations and local storage
- `os` - File system operations and path management
- `datetime` - Time handling and phase calculation
- `json` - JSON data processing and configuration
- `csv` - CSV file handling and data import/export
- `threading` - Multi-threading for background operations
- `re` - Regular expressions for text processing

### External Dependencies
```txt
# Environment and Configuration
python-dotenv==1.2.1

# AI and Machine Learning
openai>=2.0.0
scikit-learn>=1.4.0
numpy>=1.24.0

# Natural Language Processing
spacy>=3.8.0
nltk>=3.9.0

# User Interface
kivy>=2.3.0
pygame>=2.6.0

# Speech and Audio
gtts>=2.5.0
speech-recognition>=3.10.0
sounddevice>=0.5.0
pyaudio>=0.2.14
pocketsphinx>=5.0.0

# Google Services Integration
pydrive2>=1.21.0
google-api-python-client>=2.187.0
google-auth>=2.43.0
google-auth-httplib2>=0.2.1

# Image and Web Processing
pillow>=12.0.0
requests>=2.32.0
beautifulsoup4>=4.14.0

# Additional Utilities
pygame>=2.6.0
```

## 🛠️ Development & Testing

### Development Setup
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create feature branch** for your changes
4. **Install development dependencies** (same as regular installation)
5. **Make and test changes** thoroughly
6. **Submit pull request** with detailed description

### Testing Features

#### Environment Test
```bash
# Test OpenAI API configuration
python test_env.py

# Expected output:
# ✅ Environment variable loaded successfully!
# API Key starts with: sk-proj-...
```

#### Database Test
```bash
# Test database operations
python -c "
from Model.Database import Database
db = Database()
print('Database initialized successfully')
print(f'Current time phase: {db.get_time_phase()}')
"
```

#### Google Drive Test
```bash
# Test Google Drive connection
python -c "
from View.AACScreen import AACScreen
screen = AACScreen()
drive = screen.get_drive()
print('Google Drive connection successful')
"
```

#### TTS Test
```bash
# Test text-to-speech
python -c "
from gtts import gTTS
import pygame
import os

tts = gTTS('Hello, AAC system working!', lang='en')
tts.save('test.mp3')
pygame.mixer.init()
pygame.mixer.music.load('test.mp3')
pygame.mixer.music.play()
print('TTS test completed')
os.remove('test.mp3')
"
```

### Debugging & Troubleshooting

#### Common Issues and Solutions

**"Module not found" errors:**
```bash
# Ensure virtual environment is activated
myvirtual\Scripts\activate  # Windows
source myvirtual/bin/activate  # macOS/Linux

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

**OpenAI API errors:**
- Verify API key is correct in `.env` file
- Check OpenAI account has sufficient credits
- Ensure API key has DALL-E permissions
- Test with `python test_env.py`

**Google Drive authentication issues:**
- Verify `client_secrets.json` is properly configured
- Delete `View/mycreds.json` to re-authenticate
- Check Google Cloud project settings and enabled APIs
- Ensure OAuth consent screen is configured

**Database connection issues:**
- Verify `Model/Database/` directory exists
- Check file permissions for database creation
- Ensure SQLite3 is available (built into Python)

**Speech recognition problems:**
- Check microphone permissions
- Verify PyAudio installation
- Test microphone with system settings
- Install platform-specific audio drivers if needed

#### Debug Mode
Enable detailed logging by modifying the print statements in the code or adding:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security & Privacy

### Data Protection
- **API Keys**: Stored securely in environment variables, never in source code
- **User Data**: Processed locally with minimal cloud storage
- **Google Drive**: Used only for image backup with explicit user consent
- **Speech Data**: Temporary audio files automatically cleaned after processing
- **Local Storage**: SQLite database keeps user patterns private and local

### Best Practices
- **Regular Updates**: Keep dependencies and API keys current
- **Secure Storage**: Environment variables for all sensitive data
- **Access Control**: Google Drive access limited to designated app folder
- **Data Minimization**: Only necessary data collected and stored
- **Encryption**: HTTPS for all API communications

### Privacy Features
- **Offline Mode**: Full functionality without internet connection
- **Local Processing**: Speech and text processing done locally when possible
- **User Control**: Users control what data is shared and when
- **Transparent Operations**: Clear logging of all external API calls

## 🤝 Contributing

### How to Contribute
1. **Report Issues**: Use GitHub Issues for bugs and feature requests
2. **Code Contributions**: Submit pull requests with improvements
3. **Documentation**: Help improve documentation and examples
4. **Testing**: Test on different platforms and configurations
5. **Translation**: Help add multi-language support

### Development Guidelines
- **Code Style**: Follow PEP 8 Python style guidelines
- **Documentation**: Comment complex algorithms and business logic
- **Error Handling**: Comprehensive exception handling with user-friendly messages
- **Testing**: Test new features thoroughly before submission
- **Backwards Compatibility**: Maintain compatibility with existing data

### Areas Needing Help
- **Accessibility**: Improve support for users with disabilities
- **Performance**: Optimize processing speed and memory usage
- **Mobile Support**: Adapt interface for mobile devices
- **Localization**: Add support for multiple languages
- **Voice Profiles**: Custom voice generation and selection


### Third-Party Licenses
- **ARASAAC Pictograms**: Creative Commons license
- **OpenAI DALL-E**: Commercial API usage terms
- **Google Drive API**: Google API terms of service
- **Python Libraries**: Various open source licenses (see individual packages)

## 🆘 Support & Community

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Join community discussions in repository
- **Documentation**: Comprehensive inline documentation in code
- **Examples**: Sample configurations and usage examples

### Community Resources
- **Wiki**: Detailed guides and tutorials
- **FAQ**: Frequently asked questions and solutions
- **Video Tutorials**: Step-by-step setup and usage guides
- **User Forums**: Community support and best practices sharing

### Version History
- **v1.0.0** (Initial Release) - Core AAC functionality with basic TTS and pictograms
- **v1.1.0** (AI Integration) - Added DALL-E image generation and smart matching
- **v1.2.0** (Context Awareness) - Time-based and location-aware recommendations
- **v1.3.0** (Database Enhancement) - Advanced TF-IDF analysis and user learning
- **v1.4.0** (Current) - Google Drive integration and improved UI/UX

### Technical Roadmap
- **Performance Optimization** - Faster image processing and caching
- **API Rate Limiting** - Better handling of API quotas and limits
- **Offline Capabilities** - Enhanced offline mode with local AI models
- **Database Migration** - Tools for upgrading database schema
- **Plugin Architecture** - Support for third-party extensions

---

## 🌟 Acknowledgments

### Special Thanks
- **ARASAAC** - For providing high-quality pictogram resources
- **OpenAI** - For DALL-E API enabling custom image generation
- **Google** - For Drive API and cloud storage services
- **Kivy Community** - For the excellent cross-platform UI framework
- **AAC Community** - For feedback, testing, and feature suggestions

### Research & Inspiration
- Evidence-based AAC research and best practices
- Accessibility guidelines and universal design principles
- User feedback from individuals with communication needs
- Speech-language pathology professional input

---

**Made with ❤️ for the AAC community**

*This application is dedicated to empowering individuals with communication difficulties by providing accessible, intelligent, and adaptive communication tools. Our goal is to break down communication barriers and enable everyone to express themselves effectively.*

**Project Status:** Active Development  | **Platform:** Cross-platform (Windows, macOS, Linux)