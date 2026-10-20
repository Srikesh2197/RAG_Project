"""Stage 7: check the chat's follow-up rewriting on hand-written two-turn examples.

    python scripts/stage07_followups.py --estimate   # print the cost and stop
    python scripts/stage07_followups.py              # about a tenth of a cent

The corpus has no labelled conversations, so this is a spot check and not a measurement:
ten examples, written by hand about articles in the corpus, and none of it is in a
results table. Each example is an earlier turn and a new message. Eight lean on the
earlier turn; the check is that the rewritten question names what the message left out.
Two do not lean on it (a new topic, and a question complete in itself); the check is that
they come back unchanged, since pulling an old topic into a new question is the usual way
this step does harm.

The check is a string test on a model's free text, so read the output as well. The
rewriter is the one configured in configs/chat.yaml, which the app uses.
"""

import argparse
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

from ragbasics.config import PipelineConfig, load_config
from ragbasics.costs import append_ledger, generation_cost
from ragbasics.query.followup import SYSTEM_PROMPT, build_prompt
from ragbasics.registry import build

CONFIG = Path("configs/chat.yaml")
LEDGER = Path("runs/cost_ledger.csv")

# (earlier question, its answer, the new message, what the rewrite must mention).
# An empty tuple means the message should come back unchanged.
EXAMPLES = [
    ("Who is facing a criminal trial on fraud and conspiracy charges, according to TechCrunch?",
     "Sam Bankman-Fried.", "What did he plead?", ("Bankman-Fried",)),
    ("Which company removed AI-created songs and introduced an AI DJ feature?",
     "Spotify.", "Did it report an operating profit this year?", ("Spotify",)),
    ("What did Valve announce for the Steam Deck in November 2023?",
     "A new version of the Steam Deck with an OLED screen.",
     "How did Engadget review it?", ("Steam Deck", "Engadget")),
    ("What does the TechCrunch article say about GPT-4 and toxic output?",
     "That GPT-4 can be prompted into toxic output more easily than other models.",
     "And what about Meta's open source approach?", ("Meta", "TechCrunch")),
    ("How did Sporting News describe the way sportsbooks set betting lines in October 2023?",
     "They adjust lines and odds as bets come in.",
     "Was that still the case in December?", ("sportsbooks", "December")),
    ("Which company did TechCrunch report as profitable in the third quarter?",
     "Uber.", "What safety concerns has the company known about since 2014?", ("Uber", "2014")),
    ("Did the All Blacks lose to Argentina in Christchurch?",
     "Yes, Argentina beat the All Blacks in Christchurch.",
     "When did they first beat them?", ("Argentina", "All Blacks")),
    ("What did The Verge report about Donald Trump and electric vehicles?",
     "That he criticised their cost, their range and their effect on American jobs.",
     "Which of those did TechCrunch link to the North American market?", ("TechCrunch",)),
    ("Which company removed AI-created songs and introduced an AI DJ feature?",
     "Spotify.", "Who won the match between Manchester United and Bayern Munich?", ()),
    ("Who is facing a criminal trial on fraud and conspiracy charges, according to TechCrunch?",
     "Sam Bankman-Fried.",
     "What did Engadget say about the screen of the Steam Deck OLED?", ()),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    spec = load_config(CONFIG, PipelineConfig).followup
    rewriter = build("followup", spec.name, **spec.params)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_in = sum(
        len(encoding.encode(SYSTEM_PROMPT + build_prompt([(q, a)], message)))
        for q, a, message, _ in EXAMPLES
    )
    estimate = generation_cost(rewriter.model, tokens_in, 150 * len(EXAMPLES))
    print(f"{len(EXAMPLES)} calls to {rewriter.model}, about {tokens_in:,} tokens in. "
          f"Estimated cost: ${estimate:.4f} (0 for calls already cached).")
    if args.estimate:
        raise SystemExit("Estimate only: nothing was rewritten.")

    billed_in = billed_out = passed = 0
    try:
        for question, answer, message, must in EXAMPLES:
            result = rewriter.rewrite([(question, answer)], message)
            if not result.cached:
                billed_in, billed_out = (billed_in + result.input_tokens,
                                         billed_out + result.output_tokens)
            if must:
                ok = all(term.lower() in result.question.lower() for term in must)
                expect = f"must mention {', '.join(must)}"
            else:
                ok = not result.rewritten
                expect = "should come back unchanged"
            passed += ok
            print(f"\nEarlier: {question}\n         {answer}\nMessage: {message}"
                  f"\nRewrite: {result.question}\n  {'ok' if ok else 'CHECK'}: {expect}"
                  f"   ({result.input_tokens} tokens in, {result.output_tokens} out)")
    finally:
        cost = generation_cost(rewriter.model, billed_in, billed_out)
        if cost:
            append_ledger(LEDGER, "stage07:followups", rewriter.model, billed_in, billed_out, cost)
    print(f"\n{passed} of {len(EXAMPLES)} passed the string check. Billed now: ${cost:.4f}.")


if __name__ == "__main__":
    main()
