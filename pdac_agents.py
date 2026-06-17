import os
import sys
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000
SEARCH_USES = 5
OUTPUT_DIR = "./pdac_findings"
FINDINGS_TRIM = 1500
MAX_RETRIES = 6
RETRY_BASE = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

log = logging.getLogger("pdac")
log.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(fmt)
console.stream.reconfigure(encoding="utf-8", errors="replace")

logfile = logging.FileHandler(
    f"{OUTPUT_DIR}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    encoding="utf-8"
)
logfile.setLevel(logging.DEBUG)
logfile.setFormatter(fmt)

log.addHandler(console)
log.addHandler(logfile)

for noisy in ("httpx", "httpcore", "anthropic._base_client", "anthropic"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": SEARCH_USES,
}

AGENTS = {
    "kras_mutations": {
        "name": "KRAS & Oncogene Specialist",
        "mission": """You are a molecular oncology researcher specializing in KRAS mutations in pancreatic ductal adenocarcinoma (PDAC).

First, search for the current state of KRAS G12D/G12V/G12R inhibitors, synthetic lethal interactions, and resistance mechanisms.

Then — and this is the critical part — reason beyond what you find. Based on the mechanisms you discover, propose:
- A novel molecular combination that has NOT been clinically tested
- A specific structural modification to an existing KRAS inhibitor that could overcome a known resistance mechanism
- A synthetic lethal target in KRAS-mutant PDAC that nobody has published a drug against yet

Be explicit about what is known vs what you are reasoning/extrapolating. Mark your novel proposals clearly with [NEW IDEA].""",
    },

    "tumor_microenvironment": {
        "name": "Tumor Microenvironment Specialist",
        "mission": """You are a researcher specializing in the pancreatic cancer tumor microenvironment and desmoplastic stroma.

First, search for current CAF subtypes, stromal targeting attempts, and what converts cold PDAC tumors to hot ones.

Then reason beyond the literature to propose:
- A specific cell-type combination intervention not yet tested (e.g. depleting CAF subtype X while activating immune cell type Y simultaneously)
- A novel stromal reprogramming approach using drugs approved for fibrotic diseases but never combined with PDAC immunotherapy
- A biomarker-driven patient stratification strategy that would identify who most benefits from stroma disruption

Mark all novel proposals clearly with [NEW IDEA]. Be honest about which parts are extrapolation.""",
    },

    "microbiome_fungi": {
        "name": "Microbiome & Mycobiome Specialist",
        "mission": """You are a researcher studying the microbiome and mycobiome in pancreatic cancer.

First, search for the Malassezia-PDAC findings, IL-33/complement cascade work, and any antibiotic/antifungal PDAC trials.

Then reason forward to propose:
- A specific intervention protocol using existing antifungals combined with immune checkpoint therapy that has not been formally tested
- A gut microbiome modulation strategy (specific probiotic strains or FMT protocol) targeting the PDAC tumor microenvironment
- A diagnostic test using fungal or bacterial signatures to predict gemcitabine response — describe what you would measure and how

Mark all novel proposals clearly with [NEW IDEA]. Distinguish clearly between what the literature shows and what you are inventing.""",
    },

    "metabolism": {
        "name": "Cancer Metabolism Specialist",
        "mission": """You are a researcher focused on metabolic reprogramming in pancreatic cancer.

First, search for macropinocytosis inhibitors, autophagy dependence in PDAC, and lipid metabolism vulnerabilities.

Then reason beyond existing work to propose:
- A novel metabolic combination attack: two approved drugs from different metabolic pathways that together starve PDAC cells via mechanisms that have not been combined
- A specific metabolic state (nutrient conditions, metabolite levels) that would make PDAC maximally vulnerable to an existing drug
- A new molecular target in the macropinocytosis or lipid synthesis pathway with no current drug — describe what kind of molecule could inhibit it

Mark all novel proposals with [NEW IDEA]. Be specific about molecular mechanisms.""",
    },

    "drug_repurposing": {
        "name": "Drug Repurposing Specialist",
        "mission": """You are a researcher finding existing approved drugs to reposition for pancreatic cancer.

First, search for beta-blockers, antihistamines, antifungals, complement inhibitors, and other non-oncology drugs with PDAC signals.

Then go further and propose:
- A specific triple combination (one PDAC-approved drug + two repurposed drugs) with a mechanistic rationale for why all three together produce synergy beyond any pair
- A drug approved for a completely unrelated condition (autoimmune, psychiatric, metabolic) whose mechanism of action maps onto a PDAC vulnerability — explain the mapping in detail
- A clinical trial design (patient selection criteria, dosing schedule, primary endpoint) for the most promising repurposing candidate you find

Mark all novel proposals with [NEW IDEA]. The goal is actionable ideas, not literature review.""",
    },
}


def get_retry_after(e):
    try:
        retry_after = e.response.headers.get("retry-after")
        if retry_after:
            wait = int(retry_after) + 2
            log.debug(f"retry-after header says {retry_after}s, using {wait}s")
            return wait
    except Exception:
        pass
    return RETRY_BASE


def api_call_with_retry(label, fn):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.debug(f"[{label}] API call attempt {attempt}/{MAX_RETRIES}")
            result = fn()
            log.debug(f"[{label}] Success on attempt {attempt}")
            return result
        except anthropic.RateLimitError as e:
            if attempt == MAX_RETRIES:
                log.error(f"[{label}] Rate limit — giving up after {MAX_RETRIES} attempts")
                raise
            wait = get_retry_after(e)
            log.warning(f"[{label}] Rate limit hit — API says wait {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            log.error(f"[{label}] API error {e.status_code}: {e.message}")
            raise
        except Exception as e:
            log.error(f"[{label}] Unexpected error: {type(e).__name__}: {e}")
            raise


def run_agent(agent_key):
    agent = AGENTS[agent_key]
    name = agent["name"]
    log.info(f"[{name}] Starting research + novel ideation")

    prompt = agent["mission"] + """

Structure your full response as:

WHAT THE LITERATURE SHOWS:
FINDING 1: [specific finding from research]
MECHANISM: [how it works]
CURRENT DRUG/TARGET: [existing compound if any]
GAP: [what is still unknown]

FINDING 2: [repeat as needed]

NOVEL PROPOSALS (your original thinking beyond the literature):
[NEW IDEA] 1: [name your proposal]
REASONING: [step by step logic for why this should work]
SPECIFIC MOLECULES/COMPOUNDS: [be as precise as possible]
WHAT WOULD PROVE IT: [the experiment that would validate this in 6 months]

[NEW IDEA] 2: [repeat]

TOP HYPOTHESIS: Your single most original, testable idea from this session."""

    try:
        def call():
            return client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=[WEB_SEARCH_TOOL],
                messages=[{"role": "user", "content": prompt}]
            )

        response = api_call_with_retry(name, call)

        text_parts = []
        search_count = 0
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            else:
                block_type = getattr(block, "type", type(block).__name__)
                if "tool" in str(block_type).lower() and "result" not in str(block_type).lower():
                    search_count += 1
                    log.debug(f"[{name}] Web search #{search_count} executed")

        full_text = "\n".join(text_parts)
        new_ideas = full_text.count("[NEW IDEA]")

        log.info(f"[{name}] Done — {search_count} web searches, {new_ideas} novel proposals generated, {response.usage.output_tokens} output tokens")

        return {
            "agent": agent_key,
            "name": name,
            "findings": full_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "searches": search_count,
            "new_ideas": new_ideas,
            "success": True,
        }

    except Exception as e:
        log.error(f"[{name}] Failed: {type(e).__name__}: {e}")
        return {
            "agent": agent_key,
            "name": name,
            "findings": f"Error: {e}",
            "success": False,
        }


def run_synthesis(agent_results):
    log.info("[Synthesis] Building cross-domain hypothesis prompt")

    successful = [r for r in agent_results if r["success"]]
    total_new_ideas = sum(r.get("new_ideas", 0) for r in successful)
    log.info(f"[Synthesis] Synthesizing {len(successful)} agents with {total_new_ideas} total novel proposals")

    findings_text = ""
    for result in successful:
        trimmed = result["findings"][:FINDINGS_TRIM]
        log.debug(f"[Synthesis] Including {result['name']}: {len(trimmed)} chars")
        findings_text += f"\n\nFROM: {result['name']}\n{'-'*40}\n{trimmed}"

    prompt = f"""You are a senior oncology researcher reviewing novel proposals from five specialist agents studying pancreatic ductal adenocarcinoma.

Your job is NOT to summarize. Your job is to combine the [NEW IDEA] proposals across agents into cross-domain hypotheses that are more powerful than any single agent's idea alone.

Look specifically for:
1. A [NEW IDEA] from one agent that amplifies a [NEW IDEA] from another agent
2. A mechanism discovered by one agent that explains why another agent's novel proposal would work
3. A combination of proposals from 3 different agents that together constitute a complete treatment strategy

{findings_text}

Generate exactly 3 cross-domain hypotheses. Each must combine ideas from at least 2 different agents. Each must be genuinely novel — not just a restatement of existing treatments.

Format each as:

HYPOTHESIS [N]: [one sentence — what is the proposed intervention]
MECHANISM: [2-3 sentences — precisely why this should work at the molecular level]
CONNECTS: [which agent proposals this combines and how]
INTERVENTION: [specific drugs, doses, sequence if relevant — be precise]
PATIENT SELECTION: [which PDAC patients would benefit most and how to identify them]
EXPERIMENT: [the specific lab or clinical experiment to test this in under 6 months]
ESTIMATED COST: [rough order of magnitude for a pilot]
WHY NOBODY HAS DONE THIS: [honest assessment]
CONFIDENCE: [Low / Medium / High] — [one sentence justification]

End with:
PRIORITY: The single hypothesis you would fund first, and the one thing that would most quickly prove or disprove it."""

    log.info(f"[Synthesis] Prompt ready ({len(prompt)} chars, ~{len(prompt)//4} tokens estimated)")

    def call():
        return client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )

    response = api_call_with_retry("Synthesis", call)
    result = response.content[0].text

    log.info(f"[Synthesis] Done — {response.usage.input_tokens} input tokens, {response.usage.output_tokens} output tokens")
    return result


def save_results(agent_results, synthesis):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{OUTPUT_DIR}/run_{timestamp}.txt"

    log.info(f"[Save] Writing to {filepath}")

    with open(filepath, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"PDAC RESEARCH RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for result in agent_results:
            f.write(f"[{result['name'].upper()}]\n")
            f.write("-" * 40 + "\n")
            f.write(result["findings"])
            f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("CROSS-DOMAIN HYPOTHESES\n")
        f.write("=" * 70 + "\n\n")
        f.write(synthesis)

    size_kb = round(os.path.getsize(filepath) / 1024, 1)
    log.info(f"[Save] Done — {size_kb} KB written to {filepath}")
    return filepath


def run_full():
    log.info("=" * 55)
    log.info("PDAC Research Intelligence — starting run")
    log.info(f"Agents: {len(AGENTS)} | Model: {MODEL} | Searches per agent: {SEARCH_USES}")
    log.info("=" * 55)

    start = time.time()
    agent_results = []

    log.info(f"Launching {len(AGENTS)} agents in parallel...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_agent, key): key for key in AGENTS}
        for future in as_completed(futures):
            result = future.result()
            agent_results.append(result)
            done = len(agent_results)
            total = len(AGENTS)
            log.info(f"Progress: {done}/{total} agents complete")

    elapsed = round(time.time() - start, 1)
    successful = [r for r in agent_results if r["success"]]
    total_searches = sum(r.get("searches", 0) for r in successful)
    total_new_ideas = sum(r.get("new_ideas", 0) for r in successful)
    total_tokens = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in successful)

    log.info(f"All agents done in {elapsed}s")
    log.info(f"Succeeded: {len(successful)}/{len(AGENTS)}")
    log.info(f"Web searches performed: {total_searches}")
    log.info(f"Novel proposals generated: {total_new_ideas}")
    log.info(f"Total tokens used: {total_tokens:,}")

    synthesis = run_synthesis(agent_results)
    filepath = save_results(agent_results, synthesis)

    total_elapsed = round(time.time() - start, 1)
    log.info(f"Full run complete in {total_elapsed}s — results saved to {filepath}")

    print("\n" + "=" * 70)
    print("CROSS-DOMAIN HYPOTHESES")
    print("=" * 70)
    print(synthesis)
    print(f"\nFull report saved to: {filepath}")


def run_quick():
    quick_keys = ["kras_mutations", "microbiome_fungi", "drug_repurposing"]
    original = dict(AGENTS)
    AGENTS.clear()
    AGENTS.update({k: original[k] for k in quick_keys})
    log.info(f"Quick mode — 3 agents: {quick_keys}")
    run_full()
    AGENTS.clear()
    AGENTS.update(original)


def ask_question(question):
    log.info(f"Single question: '{question}'")

    def call():
        return client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[WEB_SEARCH_TOOL],
            messages=[{
                "role": "user",
                "content": f"""You are a pancreatic cancer research expert.

First search for what is currently known about this topic:
{question}

Then go beyond the literature: based on what you find, propose at least one [NEW IDEA] — 
a novel hypothesis, drug combination, or experimental approach that has not been published.
Be specific about molecules, mechanisms, and how you would test it."""
            }]
        )

    response = api_call_with_retry("Ask", call)
    log.info(f"Done — {response.usage.input_tokens} input / {response.usage.output_tokens} output tokens")

    for block in response.content:
        if hasattr(block, "text"):
            print(block.text)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set in .env file")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  py pdac_agents.py run")
        print("  py pdac_agents.py quick")
        print('  py pdac_agents.py ask "your question"')
        return

    command = sys.argv[1].lower()
    log.info(f"Command: {command}")

    if command == "run":
        run_full()
    elif command == "quick":
        run_quick()
    elif command == "ask":
        if len(sys.argv) < 3:
            print('Usage: py pdac_agents.py ask "your question"')
            return
        ask_question(" ".join(sys.argv[2:]))
    else:
        log.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()