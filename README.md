# pdac-research-agents

AI agents that research pancreatic cancer across multiple domains, then combine findings to propose novel drug combinations and treatment ideas that haven't been tried yet.

## What it does

Five specialist agents run in parallel, each focused on a different slice of PDAC research: KRAS mutations, the tumor microenvironment, the microbiome and mycobiome, cancer metabolism, and drug repurposing. Each agent searches the live web, summarizes what the literature currently shows, and then proposes original ideas beyond what's published, marked clearly as `[NEW IDEA]`.

Once all agents finish, a synthesis agent reads everything and looks for connections across domains that no single agent would see on its own. The goal is to surface specific, testable hypotheses, like a drug combination or molecular intervention, that are mechanistically plausible but haven't been formally tested.

Everything gets saved to a timestamped text file so you can track how the output evolves over time as new research publishes.

## Setup

You need an Anthropic API key and web search enabled in the Anthropic Console.

To enable web search: go to console.anthropic.com, then Settings, then enable the Web Search toggle.

Install the dependencies:

```bash
pip install anthropic python-dotenv
```

Create a `.env` file in the project folder:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

Run all 5 agents:

```bash
py pdac_agents.py run
```

Run 3 agents for a faster, cheaper test:

```bash
py pdac_agents.py quick
```

Ask a specific question:

```bash
py pdac_agents.py ask "what do we know about the complement cascade in pancreatic tumors"
```

## Output

Results are saved to `pdac_findings/run_TIMESTAMP.txt`. Each file contains the full findings from every agent followed by the cross-domain hypotheses from the synthesis agent.

The session log is also saved there if you want to review exactly what happened, how many web searches ran, and how many tokens were used.

## Rate limits

The Anthropic free tier has a 30,000 input token per minute limit. Running all 5 agents in parallel can hit this. The script handles it automatically by reading the `retry-after` header from the API response and waiting exactly as long as needed before retrying.

If you want to avoid rate limits entirely, run `quick` instead of `run`, or add credits to your Anthropic account to increase the limit.

## Project structure

```
pdac_agents.py       main script
.env                 your API key, not committed to git
pdac_findings/       output folder, not committed to git
```
