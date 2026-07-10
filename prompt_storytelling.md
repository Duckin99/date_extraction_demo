I actually like your idea **much more** than my previous outline.

The reason is that **your presentation theme is "AI Workflow in Data Science Department"**, not "Computer Vision Project." Therefore, the audience first needs to understand **where this workflow fits in the business** before you introduce the AI.

If I were designing this like a McKinsey consultant (and aligned with the PUMA material), I would first establish the **business context**, then narrow down to the specific workflow, then reveal the insight, and finally introduce the AI solution.

---

# Overall Story

**Business Context**
↓
**Specific Workflow**
↓
**Pain Point**
↓
**AI Workflow**
↓
**Business Impact**

This creates a much smoother narrative.

---

# Slide 1 — Business Context

## Title

**AI Workflow in the Data Science Department**

### Subtitle

**Case Study: AI-Assisted Travel Timeline Extraction for Insurance Underwriting**

---

## Layout

```
                 Foreign Customer

                        │

        Applies for Health Insurance

                        │

             Underwriter Reviews
        Multiple Supporting Documents

 ┌────────────┬─────────────┬──────────────┐
 │ Passport   │ Visa        │ Other Docs   │
 └────────────┴─────────────┴──────────────┘

                        │

      Verify Eligibility for Insurance
```

On the bottom-right, put a small callout:

> **~210 passport pages reviewed per month**

---

## Speaker Notes

> Let me first introduce the business context behind this project.
>
> In the underwriting process, underwriters review multiple supporting documents to assess whether an applicant is eligible for insurance.
>
> Foreign customers are also part of this process, and one of the key documents they submit is their passport.
>
> One requirement is verifying how long the customer has stayed in Thailand, as this helps determine their eligibility.
>
> This is where my project focuses.

Notice:

No AI yet.

No models.

Only business.

---

# Slide 2 — Current Workflow & Pain Point

## Layout

```
Passport

↓

Locate Thai Entry/Exit Stamps

↓

Read Entry & Exit Dates

↓

Build Travel Timeline

↓

Determine Eligibility
```

On the right

💡

Current process

• Manual

• Repetitive

• 2–3 minutes / passport

---

## Speaker Notes

> Today, this workflow is performed manually.
>
> Underwriters need to locate every Thai immigration stamp, extract the relevant dates, and build a travel timeline before making their assessment.
>
> Although each passport usually contains only one or two relevant pages, this repetitive process still takes around two to three minutes per case.

Then...

Pause.

> During discussions with the operations team, I realized something interesting.

---

Next sentence

> The underwriting decision itself isn't the repetitive part.
>
> Most of the effort is spent searching for stamps and extracting dates.

This is your insight.

---

# Slide 3 — AI Workflow

## Layout

```
Passport

↓

AI Detects Thai Stamp

↓

AI Extracts Dates

↓

AI Generates Timeline

↓

Underwriter Reviews

↓

Save
```

On the right

Demo GIF

---

## Speaker Notes

> Rather than replacing the underwriter, I redesigned the workflow to automate the repetitive steps.
>
> AI first detects Thai immigration stamps, extracts the relevant dates, and automatically generates the travel timeline.
>
> The underwriter simply reviews the result before saving it.

Then say

> This keeps the human in control while significantly reducing manual work.

---

# Slide 4 — Impact

Split screen

Current

👤

Search

Read

Type

Review

↓

AI-assisted

🤖

Detect

Extract

Prefill

👤 Review

Bottom

```
Human expertise
        +
AI automation
        =
More efficient underwriting workflow
```

---

## Speaker Notes

> The goal of this project isn't to automate underwriting.
>
> It's to automate repetitive document processing so underwriters can focus on making decisions instead of copying information.
>
> This is one example of how AI can improve workflows within the Data Science department.

---

# One thing I would improve

I would slightly refine your first slide.

Instead of saying

> Underwriters review a lot of documents every month.

I would make it more visual.

```
Insurance Application

↓

Document Review

↓

Passport ← (highlight)

↓

Travel Timeline

↓

Eligibility Decision
```

This immediately shows **where your project fits** in the overall underwriting workflow.

---

# Finally, here's the prompt I'd give NotebookLM

> You are an ex-McKinsey presentation consultant and AI workflow strategist. Help me design a 4-slide executive presentation for a 2-minute internship project showcase. The presentation theme is **"AI Workflow in the Data Science Department."**
>
> The audience includes executives, managers, mentors, and colleagues from different departments with mixed technical backgrounds.
>
> My project is an **AI-assisted travel timeline extraction workflow** for insurance underwriting.
>
> **Business context:**
>
> * Underwriters review multiple supporting documents to assess insurance eligibility.
> * Foreign applicants are also customers.
> * One important document is the passport because underwriters need to verify how long the applicant has stayed in Thailand.
> * Currently, they manually locate Thai immigration stamps, read entry and exit dates, build a travel timeline, and then make an underwriting decision.
> * Approximately 210 passport pages are reviewed each month.
> * The manual passport review takes around 2–3 minutes per passport page.
>
> **AI workflow:**
>
> * Detect Thai immigration stamps.
> * Extract dates using OCR.
> * Generate a travel timeline automatically.
> * Underwriter reviews and approves the result (human-in-the-loop).
>
> **Core message:**
> AI does not replace the underwriter. It automates repetitive document processing so underwriters can spend more time making underwriting decisions.
>
> Design the presentation in a clean McKinsey consulting style with minimal text, one key message per slide, clear process diagrams, and strong storytelling. Focus on business value rather than technical implementation. Recommend layouts, headlines, visuals, and speaker notes for each slide.

---

One last recommendation: **don't call Slide 1 "Introduction."** Give it a business headline that tells the audience why they should care. For example:

> **Insurance Decisions Begin with Document Review**

or

> **Travel History Is One Piece of the Underwriting Process**

These are much more engaging than a generic title because they immediately establish the business problem your AI workflow is addressing.
