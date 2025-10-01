
# Video Script: SupportWise AI Co-pilot Technical Design

**[Duration: 5 minutes]**

---

## Opening (0:00-0:30)

Hello! My name is Thober Detofeno, and I'm happy to share my technical design for the SupportWise AI Co-pilot challenge.

I have worked in software development for over 20 years. In the last 4 years, I've focused on AI engineering with Python, AWS, and AI models like Claude and GPT.

Let me show you how I solved this challenge and the important decisions I made.

---

## Problem Understanding (0:30-1:00)

The main challenge is helping Brenda, the Head of Support. She needs to get information from millions of Zendesk tickets from many years using normal language.

The system must handle three main tasks:
- Simple reports that return answers in less than 5 seconds
- Complex business questions that need AI analysis
- Reports that can be saved and reused to save time

---

## Architecture Overview (1:00-2:00)

My solution for this problme uses separate modules that work together and communicate through events.

For the User Interface, I use React with WebSocket connections for real-time updates. The API Gateway, built with FastAPI, manages requests and user sessions.

The center of the system is an AI Agent Platform using LangGraph, connected to an MCP Server built with FastMCP. 

For data storage, I chose PostgreSQL with TimescaleDB for time-based data, PGVector for semantic search, and Redis for temporary storage.

My idea is organizing data by age I implemented a three-tier storage strategy: hot, warm and cold data.

---

## Key Technical Solutions (2:00-3:30)

Let me explain three important technical choices:

**First: Smart Query Routing**
When Brenda asks a question, the system decides what type it ... using Gemini Flash-Lite. Simple questions use saved results or direct database queries ... with AI prompt. Complex questions... start the AI agent process. This routing saves time and money.

**Second: Multiple AI Models**
I use several AI models as backups - Claude 4 is the main one, and Gemini helps reduce costs. The system can switch between providers, so when models change ... or new ones appear, we can adapt without stopping the service.

**Third: Cost Control**
Some queries cost 1000 times more than others. The system tracks costs, sends queries... to the cheapest models, and sets limits for each user.

---

## Trade-offs and Practical Decisions (3:30-4:30)

Every system design requires choosing between different options. Here are my main choices:

**Fresh Data vs Cost**: I update data every 30 minutes instead of instantly. This reduces API costs while still meeting business needs. For data that really needs instant updates,  we can evaluate the use of Zendesk webhooks, to real-time updates.

**Speed vs Flexibility**: Pre-calculated results make common queries very fast ....but need extra storage space. I balance this by finding the 20% of queries that people use 80% of the time.

**Security vs Ease of Use**: The database connection can only read data, never change it. Even if someone tries to hack the system, they can't modify data. While this limits some features, it keeps data safe.

---

## Closing (4:30-5:00)

The system solves all technical challenges.The modular design allows improvements without breaking the service.

Most importantly, it changes how Brenda works - turning difficult data tasks ...into simple conversations, saving time and providing great insights.

Thank you for considering my application. I'm excited about the chance to being part of your team. 


---

**[End of Script]**
