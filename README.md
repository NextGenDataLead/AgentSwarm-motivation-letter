# Agent Swarm motivation letter

This swarm of agents will create a relevant and optimized motivation letter for any job-description i.c.w. your resume

## Features

- Receive the job-description through a copy-paste action (paste as one line)
- Fetch your resume (from a spreadsheet)
- Match years of expierence in job-titles, skills, sectors and tooling
- Formulate the best stratgy for the motivation letter
- Write the actual letter

## Installation
1. Make sure you have installed Anaconda/Miniconda
2. Clone the repository
3. Open the folder of the repository in your terminal
4. Run the following command to create the required conda environment with the right version of python and all dependencies installed [conda env create -f environment.yml] 

## Preparation
1. Make sure you fill out Resume/Resume_tabular.xlsx as best as possible
2. Copy the file OAI_CONFIG_LIST_example in the same folder
3. Remove the "_example" from the file
4. Open the file
5. Plug in your Chat-GPT API-key where the placeholder is now

## Run the script
1. Activate the conda environment (default = "ag_local" --> [conda activate ag_local])
2. Run the script with [python Motivation_letter.py]
3. Paste in the job-description
4. Wait for 30 seconds for the motivation letter to be ready

## Considerations
- There will be placeholders that need to be filled in, for example your name.
- Read through the output carefully as it could still be presenting false info (or it points you to a fault in the resume)
- You could always 'humanize' the text to be sure. This will prevent AI-scanners for discarding your resume even before it comes to the HR-desk.