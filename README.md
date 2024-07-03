# Aphelios Engine for Nocturio Applications
## Current version: <span style="color:blue">1.2.1+4</span>
### Written by Cameron 'mESmaC' Emery

This version is currently setup to serve recommendations to Starlight

Requirements:
  If running on a windows machine for testing, please ensure you have WSL configured.
  `vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools`


  - Redis Server
  - Python 3.9.13
  - Windows 10 SDK
  - C++ x64/x86 build tools

If there is no currently trained model follow the steps below: 

  1. py .\main.py
  2. py .\analytics.py
  3. Access the flask app interface on the localhost url provided by analyics.py
  4. Click forece fetch to start initial fetch and model training.
  5. Optionally restart the LLM after initial training

# To Note: <span style="color:red">IMPORTANT</SPAN>

This repository is not designed to be deployed as is. Certain proprietary information has been redacted. 
The main purpose this repository serves is to ensure transparency of user data collection.
