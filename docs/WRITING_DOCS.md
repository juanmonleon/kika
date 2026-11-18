# Writing the kika documentation

The rules for the documentation site (<https://kika.readthedocs.io>) outside the manual:
the **user guide**, the **API reference** and the site itself. The **manual** (the
Architecture and Methods sections, also published as the PDF) has its own rules in
`WRITING_MANUAL.md`. Four of its parts apply to every page here too: what never goes on a
page (§1.2), numbers and units (§7), citations and links (§9) and prose (§12).

These rules are not suggestions. A page that breaks one is not finished, however good it
is otherwise. To change a rule, edit this file first, in its own commit, then apply it.

The site is built by Read the Docs from `.readthedocs.yaml`. Pages are MyST Markdown
(`.md`); the existing `.rst` pages stay as they are until they are rewritten.

---

## 1. The documents and where a paragraph goes

| Section | Directory | Answers | Rules | In the PDF |
|---|---|---|---|---|
| **User guide** | `user_guide/` | *How do I do X with kika?* A goal, the calls that reach it, what comes back. | This file, §3-§5 | No |
| **Architecture** | `architecture/` | *How is kika built, and why that way?* | `WRITING_MANUAL.md` | Yes |
| **Methods** | `methods/` | *How does kika compute X, and read format X, and why that way?* | `WRITING_MANUAL.md` | Yes |
| **API reference** | `api_reference/` | *What exactly does this function take and return?* Generated from docstrings by autodoc. | This file, §6 | No |
| **About** | `about/` | What kika is, preface, licence, how to cite, glossary, known issues, troubleshooting. | Both | Partly |

Which paragraph belongs to the manual is decided by `WRITING_MANUAL.md` §1.1. In short:
it tells the reader which calls to make and in what order → **user guide**; it is the
contract of one function → **docstring**; everything that explains how kika is built or
computes → **manual**.

A user-guide page never derives anything. It links to the manual. A manual page never
walks through a workflow. It links to the user guide.

**The desktop app has its own guides**, which are not published with this site. The
library documentation does not document the app's panels, and never depends on the app's
guides.

## 2. Navigation of the site

- **The front page** opens with a table "I want to… → go to": use kika for a task → User
  guide; know how kika computes or reads something → Methods; know why kika is built this
  way → Architecture; the exact signature of a function → API reference; a result I do not
  expect → Troubleshooting and Known issues.
- **Every section has an index page**: two or three sentences on what it covers and what
  it does not, then a table of its pages with one line each. A list of links with nothing
  around it is not an index page.
- **Troubleshooting** (`about/troubleshooting.md`) is one page, organised by symptom (§5).
  There is no FAQ: it grows without order and with invented questions.
- **Headings are the navigation.** h1 is the title, h2 and h3 below it; they form the
  sidebar and every one is a link target, so make them descriptive.

## 3. User-guide pages

One task per page. The title is the task, as a gerund phrase ("Processing an evaluation
for OpenMC"). A page that turns out to cover two tasks is split.

````markdown
# <The task>

```{admonition} At a glance
:class: note
**You want:** the goal, in one sentence.
**You need:** the inputs (files, libraries, optional dependencies).
**You get:** what comes back, and where it is written.
```

## <Step, as an imperative>        ← one h2 per step: a sentence on why, the code, its output
## What can go wrong               ← symptom → cause → fix, one bullet each; "None known." otherwise
## How it works                    ← two or three sentences and the links to the manual pages
````

- **Say what the step is for** in one sentence above its code. The code shows how; the
  sentence says why the reader makes the call.
- **Every option a reader is likely to change** is named, with what it does and when it
  does nothing. The full list of parameters is the API reference's job.
- **Do not explain the method.** One sentence, then the link to its manual page.

## 4. Examples

- **Every code block in the user guide runs.** It is executed by a test, and the output
  shown under it is the output that test checks. A block that needs data the tests do not
  have says what it needs in one line above it.
- **Show what comes back**: the printed result, the first lines of the report, or the
  figure. A reader checks their run against it.
- **Real data first.** Examples use a real evaluation, named with its library and version,
  from the small tapes the tests use. A synthetic input is used only when the point is the
  mechanics of a call, and the page says so.
- **An example shows why the call is made**, not only its syntax: the question it answers
  and how to read the answer.
- **Never hide what a user will copy.** No elided arguments that the call needs; random or
  platform-dependent output is marked as such.
- A user-guide figure is the output of its example, produced by the example's own code.
  Screenshots of the app are never used here.

## 5. Troubleshooting and known issues

- **Troubleshooting** entries have a fixed shape: the symptom or the exact message → its
  cause → the fix, with the link to the manual page that explains the cause. When a
  warning is expected behaviour, the entry says so.
- **Known issues** and changed results follow `WRITING_MANUAL.md` §10: `CHANGELOG.md`
  per release, the known-issues appendix in its current state, and one `versionchanged`
  line on the affected page.

## 6. Docstrings and the API reference

- The **docstring** is the contract of one function: parameters, return value,
  exceptions, and one paragraph saying what it does. Docstrings use NumPy style (they are
  rendered by `napoleon`).
- The **manual** is the authority for the mathematics. A docstring that needs the method
  links to it (``See :doc:`/methods/<page>`.``) and does not re-derive it.
- Existing docstrings that already explain a method are not stripped in a documentation
  pass. When a manual page is written, the docstring's equations must agree with it; if
  they disagree, the code decides, and both are fixed.
- Every public function named on any page has an API entry; a role whose target does not
  exist renders as plain code, silently.

## 7. Keeping it true

- A change to a public signature updates any user-guide page that calls it, in the same
  commit. A change to an algorithm updates its manual page (`WRITING_MANUAL.md` §11).
- Pages are built with warnings as errors before they are handed over. Read the Docs
  builds the same way (`fail_on_warning` in `.readthedocs.yaml`), so a warning fails the
  published site too:

  ```bash
  # in a separate environment, in this order, as Read the Docs does
  pip install . -r docs-requirements.txt
  sphinx-build -W -b html docs/source docs/_build/html
  KIKA_DOCS_DRAFTS=1 sphinx-build -W -b html docs/source docs/_build/review
  ```

  The second build includes `drafts/` (§8); a draft page is only checked by it.

## 8. Procedure and drafts

Every page goes through the same four steps: **understand** (read the code, its tests and
what exists), **place** (§1, and the file name: lower-case, hyphenated), **write** (to its
template and rules), **verify** (every claim against the code, every example run, the
build with `-W`), then hand over with what was checked, what is marked `TODO`, and what
you were unsure about. Manual pages follow the detailed procedure of `WRITING_MANUAL.md`
§13.

A page is **published** (wired into a toctree) only after the maintainer has read it.
Until then it lives under `docs/source/drafts/`. The published build leaves `drafts/`
out. A **review build** includes it, under a "Drafts (under review)" section and with a
banner on every draft page: `KIKA_DOCS_DRAFTS=1` locally, and on Read the Docs every
version except the published `latest` and `stable` (`docs/source/conf.py`). To publish a
page, move it out of `drafts/` to its place and add it to its section's toctree in
`index.rst`, and, for a manual page, in `pdf_index.rst` too.
