import autogen
import pandas as pd
from datetime import datetime
import os


# Configuration setup
config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

# Fetch the resume data
excel_file_path = os.path.join(os.getcwd(), 'Resume', 'Resume_tabular.xlsx') # Define the relative path to the Excel file

if not os.path.exists(excel_file_path): # Verify the file path
    raise FileNotFoundError(f"The file at path {excel_file_path} was not found. Please check the path and try again.")

sheet_name = 'Jobs' # Load data from the Excel file

df = pd.read_excel(excel_file_path, sheet_name=sheet_name) # Read the Excel file

# Calculate the ranges of experiene in skills, sectors, functions and tooling. E.g. [(2015, 2017), (2013, 2014)]
def merge_date_ranges(ranges):
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = []
    current_start, current_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))
    
    # print(f"Merged ranges: {merged_ranges}")  # Debug statement
    return merged_ranges

# Calculate the actual years of experience in skills, sectors, functions and tooling
def calculate_experience(df):
    experience = {
        "skills": {},
        "sector": {},
        "function": {},
        "tools": {}
    }

    def add_experience(experience_dict, category, start_year, end_year):
        if category not in experience_dict:
            experience_dict[category] = []
        experience_dict[category].append((start_year, end_year))

    for index, job in df.iterrows():
        start_year = int(job['Start'])
        end_year = int(job['End']) if job['End'] else datetime.now().year

        # Process skills
        for skill in job['Skills'].split(','):
            skill = skill.strip()
            add_experience(experience['skills'], skill, start_year, end_year)

        # Process sector
        sector = job['Sector'].strip()
        add_experience(experience['sector'], sector, start_year, end_year)

        # Process function
        function = job['Function'].strip()
        add_experience(experience['function'], function, start_year, end_year)

        # Process tools
        for tool in job['Tools_and_technologies_used'].split(','):
            tool = tool.strip()
            add_experience(experience['tools'], tool, start_year, end_year)

    # Merge date ranges and calculate unique years of experience
    for category in experience:
        for key in experience[category]:
            # print(f"Category: {category}, Key: {key}, Ranges before merge: {experience[category][key]}")  # Debug statement
            merged_ranges = merge_date_ranges(experience[category][key])
            unique_years = sum(end - start + 1 for start, end in merged_ranges)
            experience[category][key] = unique_years
            # print(f"Category: {category}, Key: {key}, Unique years: {unique_years}")  # Debug statement

    return experience

# Calculate experience
experience = calculate_experience(df)

# Print the calculated experience
# print("Experience calculated from Excel file:")
# print(experience)

resume_details = "\n\n".join(
    f"Company: {row['Company']}\n"
    f"Sector: {row['Sector']}\n"
    f"Function: {row['Function']}\n"
    f"Job Description: {row['Job_description']}\n"
    f"Achievements: {row['Achievements']}\n"
    f"Tools and Technologies Used: {row['Tools_and_technologies_used']}\n"
    f"Skills: {row['Skills']}\n"
    f"Start: {row['Start']}\n"
    f"End: {row['End']}"
    for _, row in df.iterrows()
)


# Define tasks for agents
job_description = input("Please enter the job description: ") # This is the start of the 'conversation' where the user pastes in a job-description
match_task = "Find the matches between the job description and the calculated experience. Reply 'MATCH_DONE' at the end."
strategy_task = "develop a detailed strategy for crafting compelling motivation and cover letters. Include guidelines on purpose and audience analysis, content structuring, language use, unique value proposition, proofreading, and follow-up tactics. Ensure this strategy is adaptable across different types of applications and provides a blueprint for effective communication in various professional contexts."
motivation_letter_task = "Write a motivation letter based on the matches found and the best strategy suggested. Reply 'TERMINATE' at the end."

# Define agents
match_maker = autogen.AssistantAgent(
    name="Match_Maker",
    llm_config=llm_config,
    system_message="""
        You are a match maker. Find the matches between the job description and the calculated experience. 
        Reply 'MATCH_DONE' at the end.
        """,
)

strategy_agent = autogen.AssistantAgent(
    name="Strategy_Agent",
    llm_config=llm_config,
    system_message="""
        Strategy Agent, you are assigned to develop an advanced strategy for writing exceptional motivation and cover letters tailored to diverse applications, including job applications, university admissions, and grant proposals. Your strategy should address the following key areas:

Purpose and Audience Analysis: Devise methods for identifying the purpose of each letter, aligning it with the organization's objectives, and analyzing the intended audience to tailor the content accurately.

Content Structuring and Personalization: Outline a plan for structuring these letters and personalizing them to highlight the applicant's unique skills and passions, and the qualifications. Most impotantly always mention the relevant achievements.

Persuasive Language Usage: Provide guidance on selecting persuasive language and maintaining a professional tone to engage the reader effectively.

Highlighting Unique Value: Suggest approaches for emphasizing the applicant's unique value, highlighting particular skills or experiences that distinguish them from other candidates, focusing on specific skills or experiences that offer particular benefits to the organization.

Proofreading and Editing Best Practices: Recommend protocols for ensuring that the documents are error-free and polished.

Effective Openings and Closings: Suggest dynamic opening lines and compelling closing statements that will catch and hold the attention of the reader, make a memorable impact and urging them to take action. 

Proceed to compile this strategy into a comprehensive document that can serve as a guideline for any individual responsible for drafting these types of letters.

Reply 'STRATEGY_DONE' at the end.
        """,
)

writing_agent = autogen.AssistantAgent(
    name="Writing_Agent",
    llm_config=llm_config,
    system_message="""
        You are a professional writer. Write a motivation letter based on the matches found and the best strategy suggested.
        Reply "TERMINATE" at the end when everything is done.
        """,
)

# User proxy agent
user_proxy_auto = autogen.UserProxyAgent(
    name="User_Proxy_Auto",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },
)


# Step 1: Find matches between job description and calculated experience
# Step 2: Find the best strategy for the motivation letter
# Step 3: Write a motivation letter

# Prepare the experience data
experience_input = f"""
Skills: {experience['skills']}
Sector: {experience['sector']}
Function: {experience['function']}
Tools: {experience['tools']}
"""

# Step 1: Find matches between job description and calculated experience
match_task_with_experience = f"""
Job Description: {job_description}
Calculated Experience: {experience_input}
"""

match_results = autogen.initiate_chats(
    [
        {
            "sender": user_proxy_auto,
            "recipient": match_maker,
            "message": match_task_with_experience,
            "summary_method": "last_msg",
            "max_turns": 1,
        }
    ]
)

match_result = match_results[0].summary

# Ensure the matching process is complete
if "MATCH_DONE" not in match_result:
    raise Exception("Matching process did not complete successfully.")

# Step 2: Find the best strategy for the motivation letter
strategy_input = f"""
Job Description: {job_description}
Calculated Experience: {experience_input}
Matches found: {match_result}
Resume details: {resume_details}
"""

strategy_results = autogen.initiate_chats(
    [
        {
            "sender": user_proxy_auto,
            "recipient": strategy_agent,
            "message": strategy_task,
            "carryover": strategy_input,
            "summary_method": "last_msg",
            "max_turns": 1,
        }
    ]
)

strategy_result = strategy_results[0].summary

# Ensure the strategy generation is complete
if "STRATEGY_DONE" not in strategy_result:
    raise Exception("Strategy generation did not complete successfully.")


# Step 3: Write a motivation letter
motivation_letter_input = f"""
Matches found: {match_result}
Strategy suggested: {strategy_result}
Resume Details: {resume_details}
"""

motivation_letter_results = autogen.initiate_chats(
    [
        {
            "sender": user_proxy_auto,
            "recipient": writing_agent,
            "message": motivation_letter_task,
            "carryover": motivation_letter_input,
            "max_turns": 1,
        }
    ]
)

# motivation_letter_result = motivation_letter_results[0].summary

# # Output the motivation letter
# print("Motivation Letter:\n", motivation_letter_result)
