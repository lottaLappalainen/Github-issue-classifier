name: Scheduled Retrain

# Runs every Monday at 06:00 UTC — fetches fresh data, checks drift,
# retrains if needed. This closes the CT (Continuous Training) loop
# described in the Google MLOps Level 1 definition.
# Can also be triggered manually from the Actions tab.
on:
  schedule:
    - cron: "0 6 * * 1"   # every Monday 06:00 UTC
  workflow_dispatch:
    inputs:
      force_retrain:
        description: 'Force retrain even if no drift detected'
        required: false
        default: 'false'

env:
  PYTHON_VERSION: "3.10"

jobs:
  retrain:
    name: Drift Check + Conditional Retrain
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # ── Step 1: Fetch fresh data ───────────────────────────────────────
      - name: Fetch fresh GitHub issues (Bronze)
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python src/data/ingest.py --pages 5

      - name: Clean (Silver)
        run: python src/data/clean.py

      - name: Featurize (Gold)
        run: python src/data/featurize.py

      # ── Step 2: Run drift monitor ──────────────────────────────────────
      # Compares new Gold meta against saved baseline.
      # Writes drift_report.json with retrain_required: true/false.
      - name: Run drift monitor
        run: python src/monitoring/monitor.py --text-drift

      - name: Show drift report
        run: |
          echo "=== Drift Report ==="
          cat monitoring/drift_report.json

      # ── Step 3: Conditional retrain ────────────────────────────────────
      # retrain_trigger.py reads drift_report.json.
      # If retrain_required=true OR force_retrain input is set, it runs
      # the full training pipeline and checks the quality gate.
      # Exit code 0 = no retrain needed OR retrain succeeded.
      # Exit code 1 = retrain ran but failed quality gate (regression).
      # Exit code 2 = drift_report.json missing (monitor didn't run).
      - name: Run retrain trigger
        run: |
          if [ "${{ github.event.inputs.force_retrain }}" = "true" ]; then
            echo "Force retrain requested — running training directly"
            python src/models/train.py
            python src/models/evaluate.py
          else
            python src/monitoring/retrain_trigger.py
          fi

      # ── Step 4: Report ─────────────────────────────────────────────────
      - name: Show version comparison
        if: always()
        run: |
          echo "=== Model Version Comparison ==="
          cat monitoring/version_comparison.json 2>/dev/null || echo "No comparison available"
          echo ""
          echo "=== Current metrics.json ==="
          cat metrics.json 2>/dev/null || echo "No metrics yet"

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: scheduled-retrain-evidence
          path: |
            monitoring/drift_report.json
            monitoring/version_comparison.json
            metrics.json
          retention-days: 90