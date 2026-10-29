"""The run-design assistant: Claude Opus 5.5 helping a person plan runs.

It reads bench data through a fixed set of read-only tools and can save draft
run designs, nothing more. A person reviews a draft in the Designs view and
is the only one who can queue or start it, so a wrong answer costs a reading,
never a run.

Modules:
- client.py: the Anthropic client and the exact request. The only reader of
  the assistant's API key.
- prompt.py: the frozen system prompt and its hash.
- redact.py: keeps secrets out of everything the assistant stores or shows.
- conversation.py: append-only conversation files and the live event log.
- costs.py: what each request cost, and the daily spend book.
- engine.py: the streamed tool loop, one turn at a time.
- types.py: the tool context and outcome shared with tools.py.

Importing this package imports none of them, so the server can load the
dashboard even if the Anthropic SDK is broken.
"""
