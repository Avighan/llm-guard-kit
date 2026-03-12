"""
LLMGuard MVP Frontend
=====================
Streamlit application covering all product use cases.

Run
---
    streamlit run app/frontend.py

Environment variables
---------------------
    ANTHROPIC_API_KEY   required
    GUARD_STATE_PATH    state persistence path (default: guard_state.pkl)
    GUARD_MODEL         Claude model (default: claude-haiku-4-5-20251001)
"""

import os
import io
import json
import time
import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "LLMGuard",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
.confidence-high   { color:#28a745; font-weight:700; font-size:1.1em; }
.confidence-medium { color:#fd7e14; font-weight:700; font-size:1.1em; }
.confidence-low    { color:#dc3545; font-weight:700; font-size:1.1em; }
.risk-bar-wrap     { background:#eee; border-radius:6px; height:12px; width:100%; }
.section-label     { font-size:0.78em; color:#888; text-transform:uppercase;
                     letter-spacing:0.08em; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)


# ── Manager (singleton across sessions) ─────────────────────────────────────────

@st.cache_resource
def get_manager():
    from app.manager import GuardManager
    return GuardManager(
        api_key    = os.environ.get("ANTHROPIC_API_KEY", ""),
        model      = os.environ.get("GUARD_MODEL", "claude-haiku-4-5-20251001"),
        state_path = os.environ.get("GUARD_STATE_PATH", "guard_state.pkl"),
    )

mgr = get_manager()


# ── Sidebar ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ LLMGuard")
    st.caption("v0.1.3 · Predict · Diagnose · Repair")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Calibrate", "Query", "Error Analysis", "QARA Adapter", "Docs"],
        label_visibility="collapsed",
    )

    st.divider()
    stats = mgr.get_stats()
    fitted = stats.get("fitted", False)
    qara   = stats.get("qara_fitted", False)
    lrn    = stats.get("learning", {})

    st.markdown(f"**KNN** {'✅ Ready' if fitted else '⚠️ Not calibrated'}")
    st.markdown(f"**QARA** {'✅ Active' if qara else '○ Off'}")
    st.markdown(f"**Pool** {lrn.get('calibration_pool_size', 0)} correct · "
                f"{lrn.get('error_log_size', 0)} errors")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY not set")

    st.divider()
    st.caption("Built with llm-guard-kit")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Dashboard")

    # ── Top metrics ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries",   lrn.get("n_queries_logged", 0))
    c2.metric("Calibration pool", lrn.get("calibration_pool_size", 0))
    c3.metric("Error log",        lrn.get("error_log_size", 0))
    c4.metric("API cost",         f"${stats.get('cost_usd', 0):.4f}")

    st.divider()

    # ── Learning loop status ─────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Learning Loop")

        until_qara = lrn.get("examples_until_qara", 50)
        until_heal = lrn.get("errors_until_heal", 5)
        from app.manager import QARA_RETRAIN_THRESHOLD, ERROR_HEAL_THRESHOLD

        st.markdown('<p class="section-label">QARA re-train</p>', unsafe_allow_html=True)
        qara_progress = max(0, 1 - until_qara / QARA_RETRAIN_THRESHOLD)
        st.progress(qara_progress,
                    text=f"{QARA_RETRAIN_THRESHOLD - until_qara}/{QARA_RETRAIN_THRESHOLD} "
                         "labeled examples")

        st.markdown('<p class="section-label">Prompt Healer</p>', unsafe_allow_html=True)
        heal_progress = max(0, 1 - until_heal / ERROR_HEAL_THRESHOLD)
        st.progress(heal_progress,
                    text=f"{ERROR_HEAL_THRESHOLD - until_heal}/{ERROR_HEAL_THRESHOLD} errors")

        st.caption(
            "Progress bars fill as feedback arrives. "
            "QARA and Healer fire automatically at threshold."
        )

    with col_b:
        st.subheader("Confidence Accuracy")
        conf_acc = lrn.get("confidence_accuracy", {})
        if conf_acc:
            rows = []
            for tier, data in conf_acc.items():
                rows.append({
                    "Tier":     tier.capitalize(),
                    "Queries":  data["total"],
                    "Correct":  data["correct"],
                    "Accuracy": f"{data['accuracy']:.1%}" if data["accuracy"] is not None else "—",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption(
                "High-confidence queries should be mostly correct. "
                "Low-confidence should be mostly incorrect. "
                "If that inverts, your calibration set needs refreshing."
            )
        else:
            st.info("No feedback submitted yet. Confidence accuracy appears after "
                    "you send feedback via the Query page.")

    st.divider()

    # ── Risk threshold info ───────────────────────────────────────────────────────
    st.subheader("Risk Thresholds (auto-calibrated)")
    rt = stats.get("risk_thresholds", {})
    if rt.get("low") is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Low→Medium boundary",  f"{rt['low']:.4f}")
        col2.metric("Medium→High boundary", f"{rt['high']:.4f}")
        col3.metric("Repair tools synthesised", stats.get("n_tools", 0))
    else:
        st.info("Calibrate the guard to see risk thresholds.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: CALIBRATE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Calibrate":
    st.title("Calibrate")
    st.markdown(
        "Provide examples the LLM is known to answer correctly. "
        "This builds the KNN reference bank used for all risk scoring."
    )

    tab_manual, tab_csv, tab_consistency = st.tabs(
        ["Paste questions", "Upload CSV", "Self-consistency (no labels)"]
    )

    # ── Tab A: paste ──────────────────────────────────────────────────────────────
    with tab_manual:
        st.markdown("Paste one question per line. All are treated as **correct** examples.")
        raw = st.text_area(
            "Correct questions",
            placeholder="What is the capital of France?\nWho wrote Hamlet?\n...",
            height=200,
        )
        if st.button("Calibrate", key="cal_paste", type="primary"):
            qs = [q.strip() for q in raw.strip().splitlines() if q.strip()]
            if len(qs) < 6:
                st.error("Need at least 6 questions.")
            else:
                with st.spinner(f"Embedding {len(qs)} questions..."):
                    mgr.calibrate(qs)
                st.success(f"Calibrated on {len(qs)} questions. KNN ready.")
                st.rerun()

    # ── Tab B: CSV ────────────────────────────────────────────────────────────────
    with tab_csv:
        st.markdown(
            "Upload a CSV with columns `question` and optionally `label` (1=correct, 0=incorrect)."
        )
        f = st.file_uploader("CSV file", type="csv")
        if f:
            df = pd.read_csv(f)
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"{len(df)} rows loaded")
            if "question" not in df.columns:
                st.error("CSV must have a 'question' column.")
            else:
                labels = list(df["label"].astype(int)) if "label" in df.columns else None
                if st.button("Calibrate from CSV", type="primary"):
                    with st.spinner("Embedding..."):
                        mgr.calibrate(list(df["question"]), labels)
                    st.success(f"Calibrated on {len(df)} examples.")
                    st.rerun()

    # ── Tab C: self-consistency ───────────────────────────────────────────────────
    with tab_consistency:
        st.markdown(
            "No labels needed. The guard samples each question N times — questions "
            "where ≥80% of responses agree are treated as 'probably correct'."
        )
        raw_sc = st.text_area(
            "Questions (one per line)",
            placeholder="What is 12 × 15?\nName the tallest mountain.\n...",
            height=150,
            key="sc_qs",
        )
        n_samples = st.slider("Samples per question", 3, 10, 5)
        if st.button("Auto-calibrate", type="primary"):
            qs = [q.strip() for q in raw_sc.strip().splitlines() if q.strip()]
            if len(qs) < 10:
                st.error("Need at least 10 questions for self-consistency.")
            else:
                with st.spinner("Running self-consistency sampling (uses API)..."):
                    try:
                        mgr.guard.fit_from_consistency(qs, n_samples=n_samples)
                        # sync calibration pool
                        mgr._correct_questions = qs
                        mgr._save_state()
                        st.success("Calibrated via self-consistency.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: QUERY
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Query":
    st.title("Query")

    if not mgr.guard._fitted:
        st.warning("Guard not calibrated. Go to **Calibrate** first.")
    else:
        question = st.text_area("Your question", height=100,
                                placeholder="What year was the Eiffel Tower built?")

        with st.expander("Custom system prompt (optional)"):
            system_prompt = st.text_area("System prompt", height=80,
                                         placeholder="You are a helpful assistant...")
            system_prompt = system_prompt.strip() or None

        if st.button("Ask", type="primary", disabled=not question.strip()):
            with st.spinner("Querying..."):
                record = mgr.query(question.strip(), system_prompt)

            # Store in session for feedback
            st.session_state["last_record"] = record

        # ── Show result ───────────────────────────────────────────────────────────
        if "last_record" in st.session_state:
            rec = st.session_state["last_record"]
            st.divider()

            col_ans, col_risk = st.columns([3, 1])
            with col_ans:
                st.subheader("Answer")
                st.markdown(rec.answer)

            with col_risk:
                st.subheader("Risk")
                conf_class = f"confidence-{rec.confidence}"
                st.markdown(
                    f'<p class="{conf_class}">{rec.confidence.upper()}</p>',
                    unsafe_allow_html=True,
                )
                risk_pct = min(100, int(rec.risk_score * 300))
                color = "#28a745" if risk_pct < 35 else "#fd7e14" if risk_pct < 65 else "#dc3545"
                st.markdown(
                    f'<div class="risk-bar-wrap">'
                    f'<div style="width:{risk_pct}%;background:{color};'
                    f'height:12px;border-radius:6px;"></div></div>'
                    f'<small>score: {rec.risk_score:.4f}</small>',
                    unsafe_allow_html=True,
                )
                st.caption(f"ID: `{rec.query_id[:8]}…`")

            st.divider()

            # ── Feedback ──────────────────────────────────────────────────────────
            st.subheader("Feedback")
            st.caption("Submitting feedback trains the guard automatically.")
            col_y, col_n, col_skip = st.columns([1, 1, 3])

            with col_y:
                if st.button("✓  Correct", use_container_width=True):
                    result = mgr.feedback(rec.query_id, is_correct=True)
                    triggered = result.get("triggered", [])
                    msg = "Feedback saved."
                    if "knn_expansion" in triggered:
                        msg += " KNN pool expanded."
                    st.success(msg)
                    del st.session_state["last_record"]
                    st.rerun()

            with col_n:
                if st.button("✗  Wrong", use_container_width=True):
                    st.session_state["show_correction"] = True

            if st.session_state.get("show_correction"):
                correct_ans = st.text_input("Correct answer (optional but recommended):")
                if st.button("Submit feedback", type="primary"):
                    result = mgr.feedback(
                        rec.query_id,
                        is_correct=False,
                        correct_answer=correct_ans.strip() or None,
                    )
                    triggered = result.get("triggered", [])
                    msg = "Feedback saved."
                    if "prompt_healing" in triggered:
                        msg += " Prompt Healer fired — repair tools updated."
                    if "qara_retrain" in triggered:
                        msg += " QARA re-trained automatically."
                    st.warning(msg)
                    st.session_state.pop("show_correction", None)
                    del st.session_state["last_record"]
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Error Analysis":
    st.title("Error Analysis")

    n_errors = len(mgr._error_questions)
    st.metric("Errors logged", n_errors)

    if n_errors == 0:
        st.info("No errors logged yet. Submit 'wrong' feedback on the Query page.")
    else:
        col1, col2 = st.columns(2)

        # ── Diagnose ──────────────────────────────────────────────────────────────
        with col1:
            st.subheader("Diagnose failure clusters")
            st.caption(
                "Clusters failures into a labeled taxonomy. "
                "Read-only — does not change the guard."
            )
            if st.button("Run Diagnosis", disabled=(n_errors < 5)):
                with st.spinner("Clustering failures and generating labels..."):
                    try:
                        clusters = mgr.diagnose_now()
                        st.session_state["clusters"] = clusters
                    except Exception as e:
                        st.error(str(e))

            if "clusters" in st.session_state:
                for c in st.session_state["clusters"]:
                    with st.expander(
                        f"Cluster {c['cluster_id']} — {c['size']} failures",
                        expanded=True,
                    ):
                        st.markdown(f"**Pattern:** {c['label']}")
                        if "suggested_fix" in c:
                            st.success(f"**Suggested fix:** {c['suggested_fix']}")
                        st.markdown("**Examples:**")
                        for ex in c.get("examples", []):
                            st.markdown(f"- Q: *{ex['question'][:200]}*")
                            st.markdown(f"  Model: {ex['model_answer'][:200]}")

        # ── Heal ──────────────────────────────────────────────────────────────────
        with col2:
            st.subheader("Prompt Healer")
            st.caption(
                "Synthesises targeted repair instructions from failure patterns "
                "and auto-injects them on future queries in those error clusters."
            )
            if st.button("Run Prompt Healer", type="primary", disabled=(n_errors < 5)):
                with st.spinner("Synthesising repair tools..."):
                    try:
                        result = mgr.heal_now()
                        st.success(
                            f"Healer ran on {result['n_errors_processed']} errors. "
                            f"Repair tools updated."
                        )
                    except Exception as e:
                        st.error(str(e))

            n_tools = len(mgr.guard._tools)
            if n_tools:
                st.metric("Active repair tools", n_tools)
                with st.expander("View repair tools"):
                    for k, tool in mgr.guard._tools.items():
                        st.markdown(f"**{tool['tool_name']}** "
                                    f"(cluster {tool['cluster_idx']}, "
                                    f"{tool['cluster_size']} errors)")
                        st.markdown(f"> {tool['system_addition']}")

        st.divider()

        # ── Error log table ───────────────────────────────────────────────────────
        st.subheader("Error log")
        rows = []
        for q, a, ca in zip(
            mgr._error_questions,
            mgr._error_answers,
            mgr._error_correct_answers,
        ):
            rows.append({"Question": q[:120], "Model answer": a[:120], "Correct": ca[:80]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: QARA ADAPTER
# ════════════════════════════════════════════════════════════════════════════════
elif page == "QARA Adapter":
    st.title("QARA Adapter")
    st.markdown(
        "**QARA** (Quality-Aware Reasoning Adapter) trains a cross-domain MLP "
        "so the guard works accurately even when live queries come from a different "
        "domain than the calibration set."
    )

    col_status, col_train = st.columns([1, 2])

    with col_status:
        st.subheader("Status")
        qara_on = mgr.guard._qara is not None
        if qara_on:
            st.success("QARA adapter active")
            st.caption("All queries are scored in 64-d adapted space.")
        else:
            st.warning("QARA not trained")
            st.caption("Queries are scored in raw 384-d MiniLM space.")

        n_labeled = len(mgr._labeled_questions)
        n_correct  = sum(mgr._labeled_labels) if mgr._labeled_labels else 0
        n_incorrect = n_labeled - n_correct
        st.metric("Labeled examples", n_labeled)
        st.metric("Correct", n_correct)
        st.metric("Incorrect", n_incorrect)

        # Save adapter
        if qara_on:
            st.divider()
            save_path = st.text_input("Save adapter to:", value="qara_adapter.pkl")
            if st.button("Save adapter"):
                try:
                    mgr.guard.save_qara(save_path)
                    st.success(f"Saved to {save_path}")
                except Exception as e:
                    st.error(str(e))

    with col_train:
        st.subheader("Train QARA")
        tab_auto, tab_upload = st.tabs(
            ["From feedback history", "Upload labeled CSV"]
        )

        # ── Train from accumulated feedback ───────────────────────────────────────
        with tab_auto:
            st.markdown(
                "Uses all labeled examples accumulated from **Query → Feedback**. "
                "Requires ≥10 correct and ≥10 incorrect examples."
            )
            if n_correct < 10 or n_incorrect < 10:
                st.info(
                    f"Need ≥10 correct (have {n_correct}) and "
                    f"≥10 incorrect (have {n_incorrect}). "
                    "Keep sending feedback on the Query page."
                )
            else:
                epochs = st.slider("Training epochs", 50, 500, 200, step=50)
                if st.button("Train QARA adapter", type="primary"):
                    with st.spinner(f"Training QARA for {epochs} epochs…"):
                        try:
                            result = mgr.fit_qara_now(epochs=epochs)
                            st.success(
                                f"QARA trained: {result['n_examples']} examples, "
                                f"{result['n_correct']} correct, "
                                f"{result['n_examples'] - result['n_correct']} incorrect."
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

        # ── Train from uploaded CSVs ───────────────────────────────────────────────
        with tab_upload:
            st.markdown(
                "Upload one CSV per domain. Each must have `question` and `label` columns."
            )
            uploaded_files = st.file_uploader(
                "Domain CSVs", type="csv", accept_multiple_files=True
            )
            if uploaded_files:
                domains = []
                for f in uploaded_files:
                    df = pd.read_csv(f)
                    if "question" not in df.columns or "label" not in df.columns:
                        st.error(f"{f.name}: needs 'question' and 'label' columns")
                    else:
                        domains.append({
                            "name":      f.name.replace(".csv", ""),
                            "questions": list(df["question"]),
                            "labels":    list(df["label"].astype(int)),
                        })
                        n = len(df)
                        nc = int(df["label"].sum())
                        st.markdown(f"- **{f.name}**: {n} examples ({nc} correct, {n-nc} incorrect)")

                if domains:
                    epochs2 = st.slider("Epochs", 50, 500, 200, step=50, key="ep2")
                    if st.button("Train from uploads", type="primary"):
                        with st.spinner("Training QARA..."):
                            try:
                                result = mgr.guard.fit_qara(domains, epochs=epochs2, verbose=False)
                                mgr._save_state()
                                st.success(
                                    f"QARA trained: {result['n_examples']} examples "
                                    f"across {result['n_domains']} domains."
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

        # ── Load saved adapter ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Load saved adapter")
        load_path = st.text_input("Adapter file path:", placeholder="qara_adapter.pkl")
        if st.button("Load adapter"):
            try:
                mgr.guard.load_qara(load_path)
                mgr._save_state()
                st.success(f"Loaded adapter from {load_path}")
                st.rerun()
            except Exception as e:
                st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DOCS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Docs":
    st.title("Documentation")

    docs_path = os.path.join(os.path.dirname(__file__), "../docs/user_guide.md")
    if os.path.exists(docs_path):
        with open(docs_path) as f:
            st.markdown(f.read())
    else:
        st.info("docs/user_guide.md not found.")

    st.divider()
    st.subheader("Quick API reference")
    st.markdown("""
| Method | Description |
|--------|-------------|
| `guard.fit(questions)` | Calibrate on known-correct examples |
| `guard.fit_from_consistency(questions)` | Auto-calibrate via self-consistency |
| `guard.fit_from_execution(questions, fn)` | Auto-calibrate via verifier function |
| `guard.query(question)` | Ask question → answer + risk_score + confidence |
| `guard.diagnose(failed_qs, answers)` | Cluster failures into taxonomy |
| `guard.learn_from_errors(qs, answers, correct)` | Synthesise repair tools |
| `guard.fit_qara(domains)` | Train cross-domain adapter |
| `guard.save_qara(path)` | Persist adapter weights |
| `guard.load_qara(path)` | Load adapter + re-fit KNN |
| `guard.get_stats()` | Usage + calibration stats |
""")
