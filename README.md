# Deep Research Agent

Deep Research Agent is a multi-agent AI system that automates advanced research and report generation using OpenAI's models. This project features an asynchronous architecture with specialized agents for planning, research, report synthesis, and email delivery. The application includes a modern Gradio web interface and a CLI for research management.

![Demo GIF](demo.gif)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Cost Considerations](#cost-considerations)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Modular multi-agent workflow for automated research
- Gradio-powered web user interface
- Asynchronous processing and robust error handling
- Customizable output reports (executive summary, key findings, analysis)
- Secure environment variable management
- Resend API integration for email delivery

## Installation

Clone this repository and set up a Python environment:

git clone https://github.com/Aidin-Sahneh/deep-research-agent.git
cd deep-research-agent
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
pip install -r requirements.txt


Copy `.env.example` to `.env` and fill in your API keys.

## Usage

### Web Interface

python app.py

Then open http://127.0.0.1:7860 in your browser.

### Command Line
python deep_research_agent.py

### Python API Example
from deep_research_agent import deep_research
import asyncio

async def main():
report = await deep_research(
query="Your research question here",
send_via_email=False
)
print(report)

asyncio.run(main())


## Configuration

- Edit `.env` to set your `OPENAI_API_KEY` and `RESEND_API_KEY`.
- `deep_research_agent.py` exposes configuration variables such as search count and model selection.

## Project Structure

deep-research-agent/
├─ app.py # Gradio web interface
├─ deep_research_agent.py # Main research logic
├─ test_email.py # Resend email tester
├─ requirements.txt # Dependencies
├─ .env.example # Template for environment variables
├─ LICENSE
├─ README.md
└─ demo.gif


## Cost Considerations

- This project uses OpenAI API, billing is usage-based.
- Each research may cost $0.05–$0.15 depending on tokens used and model selected.
- Monitor your usage at https://platform.openai.com/usage

## Testing

python test_email.py # Tests Resend API key
python deep_research_agent.py # Runs a sample research workflow


## Contributing

Contributions are welcome:
1. Fork the repository
2. Create a feature branch
3. Commit your changes with descriptive messages
4. Open a pull request for review

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Author: Aidin Sahneh  
Email: aidinsahneh19@gmail.com  
GitHub: [Aidin-Sahneh](https://github.com/Aidin-Sahneh)  
Project page: https://github.com/Aidin-Sahneh/deep-research-agent
