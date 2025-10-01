# Video Script: SupportWise AI Co-pilot Technical Design

**[Duration: 5 minutes]**

---

## Opening (0:00-0:30)

Hello! My name is Thober Detofeno, and I'm happy to share my technical design for the SupportWise AI Co-pilot challenge.

I have worked in software development for over 20 years. In the last 4 years, I've focused on AI engineering with Python, AWS, and AI models like Claude and GPT. I've designed a solution that helps non-technical users work with complex data easily .

Let me show you how I solved this challenge and the decisions I made.

---

## Problem Understanding (0:30-1:00)

The main challenge is empowering Brenda, a brilliant Head of Support who isn't a data engineer, to extract insights from millions of Zendesk tickets using natural language.

The system needs to handle three critical journeys:
- Simple reporting with less 5-second responses
- Complex business insights requiring AI analysis
- Persistent, reusable reports that save time

With millions of tickets spanning years, the technical complexity is significant. But I've designed a solution for this problem.

---

## Architecture Overview (1:00-2:00)

My architecture follows a modular, event-driven design with clear separation of concerns.

At the frontend, I use React with WebSocket connections for real-time communication. The API Gateway, built with FastAPI, handles request routing and session management.

The brain of the system is an AI Agent Platform using LangGraph for orchestration, connected to an MCP Server built with FastMCP. This combination allows flexible tool execution and resource management.

For data processing, I chose PostgreSQL with TimescaleDB for time-series optimization, PGVector for semantic search, and Redis for caching. This hybrid approach ensures both speed and scalability.

The important point is Hybrid Storage Architecture, I implemented a three-tier storage strategy: hot data for the last 90 days, warm data  for the last 1 year, and cold data more that 1 year.

---

## Key Technical Solutions (2:00-3:30)

Let me highlight three critical technical decisions:

**First: Intelligent Query Routing**
When Brenda asks a question, the system classifies it in under 500 milliseconds using Gemini Flash-Lite. Simple queries hit cached results or direct SQL. Complex queries trigger the AI agent pipeline. This routing saves both time and money.

**Second: Multi-Model AI Strategy**
I implement a fallback chain with Claude 4 as primary, GPT-4 as secondary, and Gemini for cost optimization. The system includes provider abstraction, so when models deprecate or new ones emerge, we adapt without disrupting service.

**Third: Cost Management**
Query costs vary by 1000x between simple lookups and complex AI analyses. My tiered processing model tracks costs in real-time, routes queries to the most cost-effective processor, and implements per-user budgets. Cache hits cost virtually nothing, while complex analyses are batched for efficiency.

---

## Trade-offs and Pragmatic Decisions (3:30-4:30)

Every architecture involves trade-offs. Here are my key decisions:

**Data Freshness vs Cost**: I chose 30-minute synchronization intervals instead of real-time streaming. This reduces API costs by 90% while still meeting business needs. For truly real-time metrics, we can add streaming for specific high-priority data.

**Complexity vs Maintainability**: Instead of building a custom NLP engine, I leverage proven LLMs through LangChain. This means dependency on external providers, but the abstraction layer ensures we're not locked in.

**Performance vs Flexibility**: Pre-computed materialized views accelerate common queries but require storage. I balance this by identifying the top 20% of queries that represent 80% of usage.

**Security vs Usability**: Read-only database access prevents any data modification, even if SQL injection occurs. While this limits some dynamic features, it ensures data integrity - critical for production systems.

---

## Closing (4:30-5:00)

This design delivers an MVP that can go to production quickly while scaling to handle millions of tickets and hundreds of users. The modular architecture allows iterative improvements without disrupting service.

The system addresses all eight architectural challenges: from sub-5-second response times to handling technology evolution through provider abstraction.

Most importantly, it transforms Brenda's workflow - turning complex data engineering tasks into simple conversations, saving hours weekly while providing deeper insights than ever before.

Thank you for considering my approach. I'm excited about the possibility of bringing this vision to life as part of your team. I look forward to discussing how we can build this together.

---

**[End of Script]**




# Video Script: SupportWise AI Co-pilot Technical Design

**[Duration: 5 minutes]**

---

## Opening (0:00-0:30)

Hello! My name is Thober Detofeno, and I'm happy to share my technical design for the SupportWise AI Co-pilot challenge.

I have worked in software development for over 20 years. In the last 4 years, I've focused on AI engineering with Python, AWS, and AI models like Claude and GPT. I've designed a solution that helps non-technical users work with complex support data easily.

Let me show you how I solved this challenge and the important decisions I made.

---

## Problem Understanding (0:30-1:00)

The main challenge is helping Brenda, the Head of Support. She's very smart but not a data engineer. She needs to get information from millions of Zendesk tickets using normal language.

The system must handle three main tasks:
- Simple reports that return answers in less than 5 seconds
- Complex business questions that need AI analysis
- Reports that can be saved and reused to save time

With millions of tickets from many years, this is technically difficult. But I've designed a solution that makes it simple to use.

---

## Architecture Overview (1:00-2:00)

My design uses separate modules that work together and communicate through events.

For the user interface, I use React with WebSocket connections for real-time updates. The API Gateway, built with FastAPI, manages requests and user sessions.

The center of the system is an AI Agent Platform using LangGraph, connected to an MCP Server built with FastMCP. This combination allows flexible tool use and resource management.

For data storage, I chose PostgreSQL with TimescaleDB for time-based data, PGVector for meaning-based search, and Redis for temporary storage. This mixed approach gives us both speed and the ability to grow.

My key idea is organizing data by age: recent data from the last 90 days stays ready for quick access, older data uses compression, and very old data goes to archive storage - this saves both time and money.

---

## Key Technical Solutions (2:00-3:30)

Let me explain three important technical choices:

**First: Smart Query Routing**
When Brenda asks a question, the system decides what type it is in less than half a second using Gemini Flash-Lite. Simple questions use saved results or direct database queries. Complex questions start the AI agent process. This routing saves time and money.

**Second: Multiple AI Models**
I use several AI models as backups - Claude 4 is the main one, GPT-4 is the backup, and Gemini helps reduce costs. The system can switch between providers, so when models change or new ones appear, we can adapt without stopping the service.

**Third: Cost Control**
Some queries cost 1000 times more than others. My system tracks costs as they happen, sends queries to the cheapest processor that can handle them, and sets spending limits for each user. Saved results cost almost nothing, while complex analyses are grouped together for efficiency.

---

## Trade-offs and Practical Decisions (3:30-4:30)

Every system design requires choosing between different options. Here are my main choices:

**Fresh Data vs Cost**: I update data every 30 minutes instead of instantly. This reduces API costs by 90% while still meeting business needs. For data that really needs instant updates, we can add real-time updates for specific important information.

**Complexity vs Easy Maintenance**: Instead of building our own language understanding system, I use existing AI models through LangChain. This means we depend on outside providers, but we can easily switch between them if needed.

**Speed vs Flexibility**: Pre-calculated results make common queries very fast but need extra storage space. I balance this by finding the 20% of queries that people use 80% of the time.

**Security vs Ease of Use**: The database connection can only read data, never change it. Even if someone tries to hack the system, they can't modify data. While this limits some features, it keeps data safe - which is very important for real systems.

---

## Closing (4:30-5:00)

This design creates a working product that can be used right away while being able to grow to handle millions of tickets and hundreds of users. The modular design allows improvements without breaking the service.

The system solves all eight technical challenges: from answering in less than 5 seconds to adapting when AI technology changes.

Most importantly, it changes how Brenda works - turning difficult data tasks into simple conversations, saving hours every week while providing better insights than before.

Thank you for considering my solution. I'm excited about the chance to build this as part of your team. I look forward to discussing how we can create this together.

---

**[End of Script]**
