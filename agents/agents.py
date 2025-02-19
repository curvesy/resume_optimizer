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

# Initialize the model with Hyperbolic API configuration
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Using Meta-Llama model
    openai_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJub29iZXNhbmdAZ21haWwuY29tIiwiaWF0IjoxNzM4ODg3NzEwfQ.XNGj0Nwh3J9DqotbkKT8_ensnfpPS25x2yCRJ4zWufc",  # Your Hyperbolic API key
    openai_api_base="https://api.hyperbolic.xyz/v1",  # Hyperbolic base URL
    temperature=0,
    max_tokens=1024
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
        # Open file in binary mode
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text()
            except Exception as e:
                print(f"Warning: Could not extract text from page: {str(e)}")
                continue
        
        if not text:
            return "Error: Could not extract text from PDF"
        
        return text
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
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
            # Crawl job page only
            job_result = await crawler.arun(url=job_url)
            
            # Extract only the job description, requirements and tasks
            job_content = ""
            if job_result and hasattr(job_result, 'cleaned_html'):
                soup = BeautifulSoup(job_result.cleaned_html, 'html.parser')
                
                # Get main sections
                sections = []
                for section in soup.find_all(['h1', 'h2', 'p', 'ul']):
                    text = section.get_text(strip=True)
                    if text and not any(skip in text.lower() for skip in [
                        'cookie', 'datenschutz', 'agb', 'bewerben', 'interessiert',
                        'dokumente', 'ansprechperson', 'recaptcha'
                    ]):
                        sections.append(text)
                
                job_content = "\n".join(sections)

            if not job_content:
                return {"error": "Could not extract job content"}

            # Read resume
            resume_text = read_resume_pdf(resume_path)
            
            if not resume_text:
                return {"error": "Could not read resume"}

            # Prepare the analysis prompt
            analysis_prompt = f"""
            Please analyze this resume for the following job posting:

            JOB POSTING:
            {job_content}

            RESUME:
            {resume_text}

            Please provide:
            1. Analysis of match between resume and job requirements
            2. Suggested improvements to the resume
            3. Key skills to emphasize
            """

            # Call OpenAI API
            result = llm.invoke([
                SystemMessage(content="You are a professional resume optimization assistant."),
                HumanMessage(content=analysis_prompt)
            ])
            
            return {"result": result.content}
            
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}

async def main():
    job_url = "https://join.com/companies/cross-solution/13623052-software-developer-typescript?pid=e65242534431eadcb0c9"
    # Commenting out company_url since we're not using it
    # company_url = "https://join.com/companies/cross-solution/13623052-software-developer-typescript?pid=e65242534431eadcb0c9"
    resume_path = "agents/Sina korhani Shirazi - Resume (6).pdf"
    
    if not os.path.exists(resume_path):
        print(f"Error: Resume file not found at {resume_path}")
        return
        
    # Pass None or empty string for company_url
    result = await process_application(job_url, "", resume_path)
    print("Optimization Result:", result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())





