# Video Script: SupportWise AI Co-pilot Technical Design

**[Duration: 5 minutes]**

---

## Opening (0:00-0:30)

Hello! My name is Thober Detofeno, and I'm excited to share my technical design for the SupportWise AI Co-pilot challenge.

With over 20 years in software development and the last 4 years specializing in AI engineering with Python, AWS, and generative AI models like Claude and GPT, I've designed a solution that transforms how non-technical users interact with complex support data.

Let me walk you through how I approached this challenge and the key architectural decisions I made.

---

## Problem Understanding (0:30-1:00)

The core challenge is empowering Brenda, a brilliant Head of Support who isn't a data engineer, to extract insights from millions of Zendesk tickets using natural language.

The system needs to handle three critical journeys:
- Simple reporting with sub-5-second responses
- Complex business insights requiring AI analysis
- Persistent, reusable reports that save time

With millions of tickets spanning years, the technical complexity is significant. But I've designed a solution that makes this seamless.

---

## Architecture Overview (1:00-2:00)

My architecture follows a modular, event-driven design with clear separation of concerns.

At the frontend, I use React with WebSocket connections for real-time communication. The API Gateway, built with FastAPI, handles request routing and session management.

The brain of the system is an AI Agent Platform using LangGraph for orchestration, connected to an MCP Server built with FastMCP. This combination allows flexible tool execution and resource management.

For data processing, I chose PostgreSQL with TimescaleDB for time-series optimization, PGVector for semantic search, and Redis for caching. This hybrid approach ensures both speed and scalability.

The key innovation is my tiered storage strategy: hot data for the last 90 days stays fully indexed, warm data uses compression, and cold data is archived - optimizing both performance and cost.

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
