from typing import Annotated, Dict
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from crawl4ai import AsyncWebCrawler, BrowserConfig
from pypdf import PdfReader
from langgraph.types import Command
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import logging

# Load environment variables
load_dotenv()

# Get API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the model with the API key
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=openai_api_key
)

# Update the browser config
browser_config = BrowserConfig(
    headless=True,
    verbose=True,
    browser_type="chromium",
    extra_args=[  # Changed back to extra_args
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
    ],
)

async def crawl_urls(job_url: str, company_url: str) -> Dict[str, str]:
    """Crawl job and company URLs and return their content."""
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            job_result = await crawler.arun(url=job_url)
            # Commenting out company crawling
            # company_result = await crawler.arun(url=company_url)
            
            print("Job Result Type:", type(job_result))
            print("Job Result Dir:", dir(job_result))
            print("Job Result:", job_result)
            
            # Get content from the result
            job_content = getattr(job_result, 'html', '') or getattr(job_result, 'cleaned_html', '') or str(job_result)
            # Commenting out company content
            # company_content = getattr(company_result, 'html', '') or getattr(company_result, 'cleaned_html', '') or str(company_result)
            
            return {
                "job_content": job_content,
                # "company_content": company_content  # Commented out
            }
        except Exception as e:
            print(f"Crawler error details: {str(e)}")
            return {
                "error": f"Crawling error: {str(e)}"
            }

def read_resume_pdf(pdf_path: str) -> str:
    """Extract text content from resume PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@tool
def analyze_job_requirements(
    job_content: Annotated[str, "The job posting content to analyze"]
) -> str:
    """Analyze job posting content and extract key requirements."""
    analysis_prompt = """
    Analyze the following job posting and extract key requirements including:
    - Required technical skills
    - Required experience
    - Education requirements
    - Soft skills
    - Any other important qualifications
    
    Job posting:
    {job_content}
    """
    response = llm.invoke([HumanMessage(content=analysis_prompt.format(job_content=job_content))])
    return response.content

# Comment out analyze_company_info tool
# @tool
# def analyze_company_info(
#     company_content: Annotated[str, "The company information to analyze"]
# ) -> str:
#     """Analyze company information and extract key details."""
#     analysis_prompt = """
#     Analyze the following company information and extract key details including:
#     - Company culture and values
#     - Industry focus
#     - Company size and stage
#     - Key products or services
#     - Any other relevant company characteristics
    
#     Company information:
#     {company_content}
#     """
#     response = llm.invoke([HumanMessage(content=analysis_prompt.format(company_content=company_content))])
#     return response.content

@tool
def optimize_resume(
    resume_text: Annotated[str, "The resume content"],
    requirements: Annotated[str, "The analyzed job requirements"],
    # Removing company_info parameter
    # company_info: Annotated[str, "The analyzed company information"]
) -> str:
    """Optimize resume based on job requirements."""
    optimization_prompt = """
    Review the following resume and provide specific optimization suggestions based on the job requirements:
    
    Job Requirements:
    {requirements}
    
    Current Resume:
    {resume_text}
    
    Please provide:
    1. Specific content improvements
    2. Skills to emphasize
    3. Experience alignments
    4. Format suggestions
    5. Keywords to include
    """
    response = llm.invoke([HumanMessage(content=optimization_prompt.format(
        requirements=requirements,
        # company_info=company_info,  # Removed
        resume_text=resume_text
    ))])
    return response.content

# Create analyzer agent with only job requirements tool
analyzer_agent = create_react_agent(
    model=llm,
    tools=[analyze_job_requirements],
    name="analyzer",
    prompt="You are an expert analyst. Analyze job requirements."
)

# Create optimizer agent
optimizer_agent = create_react_agent(
    model=llm,
    tools=[optimize_resume],
    name="optimizer",
    prompt="You are a resume optimization expert. Suggest improvements based on job requirements."
)

# Create supervisor workflow
workflow = create_supervisor(
    [analyzer_agent, optimizer_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing a resume optimization workflow.\n"
        "1. Use analyzer_agent to analyze job requirements\n"
        "2. Use optimizer_agent to suggest resume improvements\n"
        "Coordinate the agents to optimize resumes effectively."
    ),
    output_mode="full_history"
)

# Compile the workflow
app = workflow.compile()

# Example usage
async def process_application(job_url: str, company_url: str, resume_path: str) -> Dict:
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            # Crawl job and company pages
            job_result = await crawler.arun(url=job_url)
            company_result = await crawler.arun(url=company_url)
            
            # Extract job content from the cleaned HTML
            job_content = ""
            if job_result and job_result.cleaned_html:
                # Extract relevant sections from job posting
                job_content = "\n".join([
                    section.get_text(strip=True) 
                    for section in BeautifulSoup(job_result.cleaned_html, 'html.parser').find_all(['h1', 'h2', 'p', 'li'])
                ])
            
            # Extract company content
            company_content = ""
            if company_result and company_result.cleaned_html:
                # Extract relevant sections from company page
                company_content = "\n".join([
                    section.get_text(strip=True)
                    for section in BeautifulSoup(company_result.cleaned_html, 'html.parser').find_all(['h1', 'h2', 'p'])
                ])

            # Read resume
            resume_text = ""
            if os.path.exists(resume_path):
                reader = PdfReader(resume_path)
                resume_text = "\n".join(page.extract_text() for page in reader.pages)

            # Prepare the analysis prompt
            analysis_prompt = f"""
            Please analyze and optimize this resume based on the job posting and company information:

            JOB POSTING:
            {job_content}

            COMPANY INFORMATION:
            {company_content}

            RESUME:
            {resume_text}

            Please provide:
            1. Analysis of match between resume and job requirements
            2. Suggested improvements to the resume
            3. Key skills to emphasize
            """

            # Call OpenAI API with correct message format
            result = llm.invoke([
                SystemMessage(content="You are a professional resume optimization assistant."),
                HumanMessage(content=analysis_prompt)
            ])
            
            return {"result": result.content}
            
        except Exception as e:
            logging.error(f"Error during crawling: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}

# Update the test URLs to real ones for testing
async def main():
    job_url = "https://join.com/companies/cross-solution/13623052-software-developer-typescript?pid=e65242534431eadcb0c9"  # Use a real job posting URL
    company_url = "https://join.com/companies/cross-solution/13623052-software-developer-typescript?pid=e65242534431eadcb0c9"  # Use a real company URL
    resume_path = "backend/agents/Sina korhani Shirazi - Resume (6).pdf"
    
    result = await process_application(job_url, company_url, resume_path)
    print("Optimization Result:", result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())





