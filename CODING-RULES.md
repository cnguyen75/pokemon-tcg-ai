# Colin's coding rule-book

Write code for the next reader, which is usually me in three months. Every rule below follows from these five ideas.

## Core principles

1. **Readable beats clever.** If a one-liner needs a second read, write it as three lines.
2. **Consistency beats preference.** Once a project picks a style, every file follows it. In someone else's codebase, match what's there.
3. **Names carry the what, comments carry the why.** If code needs a comment to explain what it does, fix the names or structure first.
4. **Make the happy path obvious.** Handle edge cases early and get out of the way.
5. **Delete freely.** Dead code, stale comments, and unused variables cost more than they save. Git remembers.

## Layout and whitespace

Opening braces go on the same line; closing braces get their own line. Everything else here exists so files look the same without a formatter.

**Braces (JS/TS, Java, Kotlin, C-style languages)**

- Opening brace on the same line as the statement: `function f() {`, `if (x) {`, `} else {`.
- Closing brace on its own line, aligned with the line that opened it.
- Always use braces, even for one-line `if` bodies.

```javascript
function getRate(user) {
    if (!user) {
        return DEFAULT_RATE;
    }

    return user.rate;
}
```

**Indentation and line length**

- 4 spaces per level, never tabs. (2 spaces only if a project already uses it.)
- Max 100 characters per line. Break long calls after an opening bracket, one argument per line, with a trailing comma.
- Python strings: double quotes. JS/TS: single quotes. Pick once per project and never mix.

```python
result = compute_score(
    borrower_income,
    monthly_debt,
    credit_score,
)
```

**Blank lines**

- Two blank lines between top-level functions and classes (Python); one in brace languages.
- One blank line between methods inside a class.
- Inside a function, one blank line separates logical steps. No blank line right after an opening brace or `def`.
- Never two blank lines in a row inside a function.

**Spacing**

- One space around operators and after commas: `x = a + b`, `f(a, b)`.
- No space before a call's parenthesis: `f(x)`, not `f (x)`.
- Python keyword defaults take no spaces: `def f(x, retries=3)`. With a type hint they do: `retries: int = 3`.
- Files end with exactly one newline; no trailing whitespace.

## Naming

A name should tell the reader what the thing is without looking at its definition. Casing follows the language; the rules after the table apply everywhere.

| Thing | Python | JS/TS | Example |
|---|---|---|---|
| Variable, function | snake_case | camelCase | `monthly_income` / `monthlyIncome` |
| Class, type | PascalCase | PascalCase | `LoanApplication` |
| Constant | UPPER_SNAKE | UPPER_SNAKE | `MAX_RETRIES` |
| Private helper | `_leading_underscore` | no prefix, not exported | `_parse_row` |
| File | snake_case.py | kebab-case.ts | `loan_scoring.py` |

- **Functions are verbs:** `fetch_user`, `compute_dti`, `send_reminder`. Not `user_data` or `processing`.
- **Booleans read as yes/no questions:** `is_valid`, `has_coborrower`, `should_retry`. Never `flag` or `status`.
- **Collections are plural:** `users`, `scores_by_id`. A single item is singular: `for user in users`.
- **Put units in the name** when there's any doubt: `timeout_s`, `amount_cents`, `delay_ms`.
- **No abbreviations** except universal ones (`id`, `url`, `db`, `i` for a loop index). `cnt`, `usr`, `tmp2` are out.
- **Length matches scope.** A 3-line loop can use `i`; anything that lives longer gets a real name.
- **Don't encode the type:** `users`, not `user_list` or `users_dict`.
- **One word per concept.** If it's `fetch_` in one file, it isn't `get_` or `load_` in another for the same kind of operation.

## Functions and control flow

Each function does one job and keeps its main path at the left margin.

- **One job per function.** If describing it needs the word "and", split it.
- **Aim for under ~30 lines.** Past 50, look for a piece to extract and name.
- **Return early.** Handle invalid input and edge cases first, then write the main logic unindented.
- **Max 2 to 3 levels of nesting.** Deeper than that, extract a helper or invert a condition.
- **At most 3 to 4 parameters.** More than that, group them into a dataclass/object.
- **No boolean flag parameters** that switch behavior (`save(user, True)`). Write two functions, or use a keyword-only argument with a clear name.
- **No hidden side effects.** A function named `compute_` or `get_` doesn't write to disk, the network, or a global.
- **Positive conditions:** `if is_valid:` reads better than `if not is_invalid:`.

```python
# Avoid: the real work is buried three levels deep
def process(app):
    if app is not None:
        if app.is_complete:
            if app.income > 0:
                return score(app)
    return None

# Prefer: edge cases first, main path last
def process(app):
    if app is None or not app.is_complete:
        return None
    if app.income <= 0:
        return None

    return score(app)
```

## Data, errors, and types

Make values explicit and failures loud.

**Values**

- **No magic numbers or strings.** Name them as constants at the top of the file: `MAX_RETRIES = 3`, `DTI_LIMIT = 0.43`. `0`, `1`, and `-1` used as indexes or counters are fine.
- **Use enums** (or a fixed set of constants) for anything with a known set of values, instead of bare strings like `"approved"`.
- **Don't mutate inputs.** Return a new value instead of changing a list or dict that was passed in.
- **No mutable default arguments** in Python: use `items=None`, then `items = items or []`.

**Types**

- Python: type hints on every function signature, including the return type. Use `X | None` when a value can be missing.
- JS/TS: no `any`. Use `unknown` and narrow it if the type is genuinely unknown.
- Use dataclasses (Python) or interfaces/types (TS) for structured data instead of loose dicts passed between functions.

**Errors**

- **Raise specific exceptions:** `ValueError`, `FileNotFoundError`, or a custom class. Never a bare `Exception`.
- **Never swallow errors.** No bare `except:` or `except Exception: pass`. If you catch something, handle it, log it, or re-raise it.
- **Catch narrowly,** around the one call that can fail, not the whole function.
- **Error messages include the value that failed:** `raise ValueError(f"income must be positive, got {income}")`.
- **Fail early.** Validate inputs at the top of a function, not halfway through.

## File and project structure

Every file reads top to bottom in the same order, so I always know where to look.

**Order within a file**

1. Module docstring (only if the file's purpose isn't obvious from its name)
2. Imports
3. Constants
4. Types: classes, dataclasses, enums
5. Public functions, main entry point first
6. Private helpers (`_name`)
7. `if __name__ == "__main__":` block, if any

**Imports**

- Three groups separated by a blank line: standard library, third-party, then my own modules.
- Alphabetical within each group.
- No wildcard imports (`from x import *`). No unused imports.

**Size and scope**

- One topic per file. If a file passes ~400 lines, split it along a natural seam.
- Search before building: reuse an existing helper instead of writing a second one under a new name.
- No wrapper functions that only pass a call through, and no abstraction used exactly once.
- When new code replaces old code, delete the old path completely. No leftover branches or shims.
- Remove debug prints and temporary code before committing.

## Commenting rules

**No comment is the default.** A comment stays only if it tells the reader something the names, types, and code can't.

**A comment is allowed only for:**

1. **Non-obvious contracts.** Behavior a caller couldn't guess from the name and signature: "passing `None` clears the field," "writes partially on error," "caller must close the returned file." Being public is not enough on its own.
2. **Local hazards.** A concrete trap at this exact line that's easy to miss. Two lines max.
3. **Why-comments (small projects only).** The reason for a choice when it can't be recovered from the code: a workaround, a magic value's origin, an ordering constraint. Two lines max. In a codebase with design docs or a knowledge base, the why goes there instead.
4. **Tooling or legal text.** Lint directives, license headers, generated-file markers.

**Always delete:**

- **Narration:** anything that restates the next line, explains syntax, or walks through a function step by step. Delete it; don't shorten it.
- **Provenance:** ticket numbers, PR links, dates, names, "previously / used to / changed from." Git already has this.
- **Commented-out code.**
- **Comments made untrue** by the change I'm making, wherever they sit.

**Docstrings** follow the same bar. Write one when the function has a non-obvious contract; skip it when the name and signature already say everything. When I do write one, use Google style:

```python
def apply_payment(loan: Loan, amount_cents: int) -> Loan:
    """Returns a new Loan with the payment applied.

    Overpayment is applied to principal, not held as credit.

    Raises:
        ValueError: If amount_cents is not positive.
    """
```

**TODOs** need a concrete action and a removal condition: `# TODO: drop this fallback once all rows have user_id set.` Never a bare `# TODO: fix`.

**Tests** get no comments. The test name carries the scenario: `test_overpayment_reduces_principal`.

**Before writing a comment, try first** to rename the variable, extract a well-named function, or replace a magic number with a named constant. If that says it, no comment needed.

```python
# Avoid
# Loop over the attempts and try to fetch the document
for attempt in range(MAX_RETRIES):
    # Call the API (changed from v1 per ENG-412)
    result = client.fetch(doc_id)

# Keep
for attempt in range(MAX_RETRIES):
    # Vendor rate-limits per 10s window; shorter backoff just burns attempts.
    time.sleep(10 * attempt)
    result = client.fetch(doc_id)
```

## Writing for repos (READMEs and docs)

A README is read by someone skimming from my website, often a recruiter or engineer with a minute to spare. The first two lines decide whether they keep reading, so the top stays tight; the depth goes further down, where explaining more is fine.

**README structure**

1. **One-line summary:** what it is and what it does, in plain words. No tagline.
2. **Result or status:** the headline outcome with a real number, or an honest status ("in progress," "negative result").
3. **Demo:** a screenshot, GIF, or live link, if one exists.
4. **How it works:** the approach and the key design decisions, with why. This is where detail belongs.
5. **Run it:** copy-paste commands that actually work from a fresh clone.
6. **Limitations and next steps:** what it doesn't do and what I'd change.

**Content rules**

- **Specific over impressive.** "Cut peak training memory by N% with gradient checkpointing" (with the real N), not "significantly optimized performance." If there's no number, describe what it does.
- **Honest scope.** Say if it's a side project, a course project, or a negative result. A clear negative result reads better than an oversold one.
- **Explain decisions, not just steps.** "Used X because Y" is what shows judgment. A list of tools used doesn't.
- **First person is fine:** "I built," "I chose." No "we" for solo work.
- **Nothing that isn't cleared to publish:** no employer internals, no data or figures still under review, no credentials or private paths.
- **No dated claims that go stale.** Avoid "as of" text and "currently the best" unless I'll keep it updated.

**Banned patterns** (AI tells, and weak writing in general)

- No em dashes.
- No "It's not X, it's Y" or "not just X, but Y."
- No groups of three unless there are actually three things.
- No marketing words: powerful, seamless, robust, cutting-edge, leverage, blazing fast, revolutionary.
- No "delve," "tapestry," "landscape," "pivotal," "testament."
- No announcing ("It's worth noting that"). Just say it.
- No closing summary that repeats the README.
- Headings say what the section contains ("How it works"), never tease ("The Secret Sauce").
- No emojis in headings; bold only for the occasional key term.

**Mechanics**

- Sentence case for headings.
- Always use the Oxford comma: "tests, docs, and examples."
- Numerals for measurements and anything 10 or above; units always written out or abbreviated consistently (`ms`, `GB`).
- Code, file names, and commands in `backticks`; anything longer than one line in a code block.

**My voice:** explaining a bit more than strictly needed is fine in "How it works." Padding isn't: no restating a point, no summary of what was just said.

## Pre-commit checklist

Read the whole diff once as a reviewer trying to reject it, then tick through this list.

- [ ] Every name says what the thing is; booleans read as questions; units are in names where needed
- [ ] No function over ~30 lines without a reason; nothing nested more than 3 levels
- [ ] No magic numbers or strings; constants are named at the top
- [ ] Every error is handled, wrapped with context and re-raised, or logged as final. Nothing swallowed
- [ ] Type hints on every signature; no `any`
- [ ] No narration, provenance, or commented-out code; every remaining comment passes the bar
- [ ] Comments near changed code are still true
- [ ] TODOs have an action and a removal condition
- [ ] No debug prints, unused imports, or leftovers from an abandoned approach
- [ ] Old code this replaces is fully deleted
- [ ] Braces, indentation, blank lines, and quotes match the rest of the file. If a README changed, it follows the repo-writing rules
