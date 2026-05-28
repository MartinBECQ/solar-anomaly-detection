## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


## 6. Environment

**Runtime**
- Python 3.14
- OS: macOS
- IDE: VSCode + integrated terminal
- Claude Code launched from VSCode terminal



**Global conventions (all projects)**
- All comments and docstrings in English
- Type hints on all function signatures
- No hardcoded file paths — pass filepath as parameter
- Scaler always fitted on train data only, never on full dataset
- Each project has its own BACKLOG.md at root



### Project 1 — Solar Inverter Anomaly Detection
**Goal:** detect underperforming inverters using unsupervised ML
**Dataset:** Kaggle — Solar Power Generation Data
**ML:** Isolation Forest
**Viz:** Streamlit + Plotly
**Key decisions:**
- contamination=0.05 (adjust after visual validation)
- Nighttime filter: IRRADIATION > 0.01
- Features: DC_POWER, perf_ratio (DC_POWER/IRRADIATION), power_deviation

```
solar-anomaly-detection/
├── CLAUDE.md
├── BACKLOG.md
├── data/raw/               ← CSVs from Kaggle, never modified
├── data/processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── visualization.py
├── dashboard/app.py
├── models/
└── notebooks/01_eda.ipynb
```
