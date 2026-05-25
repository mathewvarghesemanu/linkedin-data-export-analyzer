import io
import json
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import nltk
import ollama
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ── Load config ────────────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).parent / "config" / "settings.json"
CONFIG = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(CONFIG.get("data_dir", "data"))
CACHE_DIR = Path(CONFIG.get("cache_dir", "cache"))
DEFAULT_MODEL = CONFIG.get("model", "gemma4:latest")
TOPICS = CONFIG.get("topics", [])

COLOR_MAP = {"Original Post": "#0A66C2", "Reshared Post": "#F5A623"}
CATEGORY_ORDER = ["Original Post", "Reshared Post"]

EXTRA_STOPWORDS = {
    "will", "one", "us", "also", "even", "get", "got", "much", "many",
    "make", "made", "way", "like", "just", "know", "think", "time",
    "really", "very", "can", "use", "used", "using", "want", "need",
    "people", "good", "great", "new", "work", "well", "still", "lot",
    "years", "year", "something", "things", "thing", "going", "come",
    "see", "look", "take", "first", "last", "may", "would", "could",
    "never", "always", "every", "back", "put", "right",
}

# ── Data helpers ───────────────────────────────────────────────────────────────


def find_zip() -> Path | None:
    zips = sorted(
        p for p in DATA_DIR.glob("Complete_*.zip")
        if not p.name.endswith(".zip.zip")
    )
    return zips[-1] if zips else None


def extract_upload(uploaded_file) -> Path:
    # Folder keeps the .zip name so find_zip()'s `Complete_*.zip` glob still matches it.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / uploaded_file.name
    if dest.exists():
        return dest
    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zf:
        zf.extractall(dest)
    return dest


def safe_name(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "_")
    return cleaned if cleaned else "default"


def cache_path(name: str, kind: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{safe_name(name)}_{kind}_cache.json"


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


@st.cache_data(show_spinner=False)
def load_shares(zip_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{zip_path}/Shares.csv",
        usecols=["Date", "SharedUrl", "ShareCommentary"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["post_type"] = df["SharedUrl"].fillna("").str.strip().apply(
        lambda x: "Reshared Post" if x else "Original Post"
    )
    return df


@st.cache_resource
def ensure_nltk() -> None:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("stopwords", quiet=True)


@st.cache_data(show_spinner=False)
def build_word_cloud_image(text: str) -> tuple[WordCloud, list[tuple[str, int]]]:
    all_stops = STOPWORDS | EXTRA_STOPWORDS
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        stopwords=all_stops,
        max_words=150,
        collocations=False,
    ).generate(text)
    words = [
        w.lower() for w in text.split()
        if w.isalpha() and len(w) > 2 and w.lower() not in {s.lower() for s in all_stops}
    ]
    top20 = Counter(words).most_common(20)
    return wc, top20


@st.cache_data(show_spinner=False)
def build_tech_tokens(text: str) -> tuple[WordCloud, list[tuple[str, int]]]:
    common_words = set(nltk_stopwords.words("english")) | EXTRA_STOPWORDS | {
        "im", "ive", "dont", "cant", "its", "thats", "youre", "weve",
        "heres", "isnt", "doesnt", "wasnt", "arent", "hadnt",
    }
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tech_tokens = [
        word for word, tag in tagged
        if tag in ("NN", "NNS", "NNP", "NNPS")
        and word.isalpha()
        and len(word) > 2
        and word.lower() not in common_words
    ]
    tech_freq = Counter(tech_tokens)
    tech_text = " ".join(tech_tokens)
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="viridis",
        stopwords=common_words,
        max_words=120,
        collocations=False,
    ).generate(tech_text)
    return wc, tech_freq.most_common(20)


def add_total_annotations(fig, totals) -> None:
    for x, y in totals.items():
        fig.add_annotation(
            x=x, y=y, text=str(y),
            showarrow=False, yanchor="bottom",
            yshift=4, font=dict(size=11),
        )


# ── Ollama helpers ─────────────────────────────────────────────────────────────


def get_installed_models() -> list[str]:
    try:
        return sorted(m.model for m in ollama.list().models)
    except Exception:
        return []


def pull_model_background(model_name: str) -> None:
    if "downloads" not in st.session_state:
        st.session_state.downloads = {}
    st.session_state.downloads[model_name] = {
        "progress": 0.0,
        "status": "downloading",
        "status_text": "Starting…",
    }

    def _pull() -> None:
        try:
            for chunk in ollama.pull(model_name, stream=True):
                completed = getattr(chunk, "completed", None) or 0
                total = getattr(chunk, "total", None) or 0
                if total > 0:
                    st.session_state.downloads[model_name]["progress"] = completed / total
                st.session_state.downloads[model_name]["status_text"] = (
                    getattr(chunk, "status", "") or ""
                )
            st.session_state.downloads[model_name]["status"] = "done"
        except Exception as e:
            st.session_state.downloads[model_name]["status"] = "error"
            st.session_state.downloads[model_name]["error"] = str(e)

    threading.Thread(target=_pull, daemon=True).start()


# ── Download progress panel (auto-reruns every 1 s while visible) ──────────────


@st.fragment(run_every=1)
def download_progress_panel() -> None:
    downloads = st.session_state.get("downloads", {})
    if not downloads:
        return
    for model, info in list(downloads.items()):
        status = info["status"]
        if status == "downloading":
            pct = info["progress"]
            st.progress(
                pct,
                text=f"⬇️ {model}: {info['status_text']} ({pct * 100:.0f}%)",
            )
        elif status == "done":
            col_msg, col_x = st.columns([6, 1])
            with col_msg:
                st.success(f"✓ {model} ready")
            with col_x:
                if st.button("✕", key=f"clr_done_{model}"):
                    del st.session_state.downloads[model]
                    st.rerun()
        elif status == "error":
            col_msg, col_x = st.columns([6, 1])
            with col_msg:
                st.error(f"✗ {model}: {info.get('error', 'unknown error')}")
            with col_x:
                if st.button("✕", key=f"clr_err_{model}"):
                    del st.session_state.downloads[model]
                    st.rerun()


# ── LLM calls ─────────────────────────────────────────────────────────────────


def classify_topic(text: str, model: str) -> str:
    prompt = (
        f"Classify this LinkedIn post into exactly one of these topics:\n"
        f"{', '.join(TOPICS)}\n\n"
        f"Post:\n{text[:1000]}\n\n"
        f"Reply with only the topic name, nothing else."
    )
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    result = resp.message.content.strip()
    return result if result in TOPICS else "Other"


def extract_companies(text: str, model: str) -> list[str]:
    prompt = (
        "Extract all company and organization names mentioned in this LinkedIn post.\n"
        'Return a JSON array of strings, e.g. ["Google", "OpenAI"].\n'
        "If none found, return []. Return ONLY the JSON array, nothing else.\n\n"
        f"Post:\n{text[:1000]}"
    )
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    raw = resp.message.content.strip()
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


# ── App ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="LinkedIn Analytics", layout="wide")
    st.title("LinkedIn Post Analytics")

    ensure_nltk()

    if "downloads" not in st.session_state:
        st.session_state.downloads = {}
    if "model_input" not in st.session_state:
        st.session_state.model_input = DEFAULT_MODEL
    # Apply model selected via installed-model buttons (must happen before widget renders)
    if "pending_model" in st.session_state:
        st.session_state.model_input = st.session_state.pop("pending_model")

    installed_models = get_installed_models()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        user_name = st.text_input(
            "Your name",
            placeholder="e.g. Jane Smith",
            help=(
                "Scopes the LLM classification cache. "
                "Each unique name gets its own cache files — "
                "results are never mixed between users."
            ),
        )
        sname = safe_name(user_name)
        st.caption(f"Cache prefix: `{sname}`")

        st.divider()

        # ── Data upload ──────────────────────────────────────────────────────────
        st.subheader("Data")
        uploaded_zip = st.file_uploader(
            "LinkedIn export (.zip)",
            type="zip",
            help=(
                "Upload your Complete_LinkedInDataExport_*.zip. "
                "Extracted into data/ on first upload; re-uploading the same "
                "file reuses the existing extraction instead of unzipping again."
            ),
        )

        st.divider()

        # ── Model selection ────────────────────────────────────────────────────
        st.subheader("LLM Model")

        downloading_set = {
            k for k, v in st.session_state.downloads.items()
            if v["status"] == "downloading"
        }

        col_inp, col_btn = st.columns([4, 1])
        with col_inp:
            current_model = st.text_input(
                "Model name",
                key="model_input",
                placeholder=DEFAULT_MODEL,
                help="Type any Ollama model name, or click an installed model below.",
            )
        with col_btn:
            st.write("")  # vertical spacer to align with text input
            model_key = (current_model or "").strip()
            if model_key and model_key not in installed_models and model_key not in downloading_set:
                if st.button("⬇️", key="dl_btn", help=f"Download {model_key}"):
                    pull_model_background(model_key)
                    st.rerun()
            elif model_key in downloading_set:
                st.write("⏳")

        # Auto-refreshing progress bars (1 s polling via st.fragment)
        download_progress_panel()

        # Installed models list with select + delete
        if installed_models:
            st.caption("Installed — click name to select:")
            for m in installed_models:
                col_m, col_del = st.columns([5, 1])
                with col_m:
                    if st.button(m, key=f"sel_{m}", use_container_width=True):
                        st.session_state.pending_model = m
                        st.rerun()
                with col_del:
                    if st.button("✕", key=f"del_{m}", help=f"Remove {m}"):
                        try:
                            ollama.delete(m)
                        except Exception as exc:
                            st.error(str(exc))
                        st.rerun()
        else:
            st.caption("No models installed yet.")

        st.divider()

        # ── Chart toggles ──────────────────────────────────────────────────────
        st.header("Charts")

        active_model = current_model or DEFAULT_MODEL

        show_monthly = st.checkbox(
            "Monthly Posts",
            value=True,
            help=(
                "Stacked bar chart of posts per month, "
                "split by Original Post vs Reshared Post."
            ),
        )
        show_yearly = st.checkbox(
            "Yearly Posts",
            value=True,
            help=(
                "Same data aggregated by year — useful for spotting "
                "long-term trends in posting frequency."
            ),
        )
        show_wordcloud = st.checkbox(
            "Word Cloud",
            value=True,
            help=(
                "Visual frequency map of all words across post commentary. "
                "Common filler words are filtered. Includes a top-20 word accordion."
            ),
        )
        show_tech_cloud = st.checkbox(
            "Technical Terms Cloud",
            value=True,
            help=(
                "Word cloud restricted to nouns and proper nouns (NLTK POS tagging). "
                "Surfaces domain jargon, tools, and named entities. "
                "Includes a top-20 term accordion."
            ),
        )
        show_topics = st.checkbox(
            "Topic Classification",
            value=True,
            help=(
                f"Each post classified into one of 11 topic buckets via LLM ({active_model}). "
                "Results cached per user — instant on subsequent runs."
            ),
        )
        show_companies = st.checkbox(
            "Company Mentions",
            value=True,
            help=(
                f"Top 30 companies/organisations extracted from posts via LLM ({active_model}). "
                "Cached per user."
            ),
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    zip_file = None
    if uploaded_zip is not None:
        with st.spinner("Preparing uploaded data…"):
            zip_file = extract_upload(uploaded_zip)
    if zip_file is None:
        zip_file = find_zip()
    if zip_file is None:
        st.error(
            "No LinkedIn export found. Upload a `.zip` in the sidebar, "
            "or place an extracted `Complete_*.zip` folder in `data/`."
        )
        return
    if not (zip_file / "Shares.csv").exists():
        st.error(f"`Shares.csv` not found in `{zip_file.name}` — is this a Complete export?")
        return

    with st.spinner("Loading Shares.csv…"):
        df = load_shares(str(zip_file))

    total_posts = len(df)
    st.caption(
        f"Source: `{zip_file.name}` · {total_posts} posts · "
        f"cache user: **{sname}** · model: **{active_model}**"
    )

    df_text = df["ShareCommentary"].dropna()
    full_text = " ".join(df_text.astype(str))

    # ── Monthly ───────────────────────────────────────────────────────────────
    if show_monthly:
        st.subheader("Posts per Month")
        monthly = (
            df.assign(month=df["Date"].dt.to_period("M").apply(lambda p: p.start_time))
            .groupby(["month", "post_type"])
            .size()
            .reset_index(name="posts")
        )
        monthly_totals = monthly.groupby("month")["posts"].sum()
        fig = px.bar(
            monthly, x="month", y="posts", color="post_type",
            barmode="stack",
            title=f"LinkedIn Posts per Month (Total: {total_posts})",
            color_discrete_map=COLOR_MAP,
            category_orders={"post_type": CATEGORY_ORDER},
        )
        add_total_annotations(fig, monthly_totals)
        fig.update_layout(
            xaxis=dict(title="Month", tickformat="%b %Y"),
            yaxis_title="Number of Posts",
            bargap=0.2,
            legend_title="Post Type",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Yearly ────────────────────────────────────────────────────────────────
    if show_yearly:
        st.subheader("Posts per Year")
        yearly = (
            df.assign(year=df["Date"].dt.year)
            .groupby(["year", "post_type"])
            .size()
            .reset_index(name="posts")
        )
        yearly_totals = yearly.groupby("year")["posts"].sum()
        fig = px.bar(
            yearly, x="year", y="posts", color="post_type",
            barmode="stack",
            title="LinkedIn Posts per Year",
            color_discrete_map=COLOR_MAP,
            category_orders={"post_type": CATEGORY_ORDER},
        )
        add_total_annotations(fig, yearly_totals)
        fig.update_layout(
            xaxis=dict(title="Year", tickmode="linear", dtick=1),
            yaxis_title="Number of Posts",
            bargap=0.2,
            legend_title="Post Type",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Word cloud ────────────────────────────────────────────────────────────
    if show_wordcloud:
        st.subheader("Word Cloud")
        with st.spinner("Generating word cloud…"):
            wc, top20_words = build_word_cloud_image(full_text)
        fig_wc, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("LinkedIn Post Word Cloud", fontsize=18)
        plt.tight_layout()
        st.pyplot(fig_wc)
        plt.close(fig_wc)
        with st.expander("Top 20 words"):
            for word, count in top20_words:
                st.write(f"**{word}**: {count}")

    # ── Technical terms cloud ─────────────────────────────────────────────────
    if show_tech_cloud:
        st.subheader("Technical Terms Cloud")
        with st.spinner("Tokenizing and tagging parts of speech…"):
            wc_tech, top20_tech = build_tech_tokens(full_text)
        fig_tech, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(wc_tech, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Technical Terms — LinkedIn Posts", fontsize=18)
        plt.tight_layout()
        st.pyplot(fig_tech)
        plt.close(fig_tech)
        with st.expander("Top 20 technical terms"):
            for word, count in top20_tech:
                st.write(f"**{word}**: {count}")

    # ── Topic classification ──────────────────────────────────────────────────
    if show_topics:
        st.subheader("Topic Classification")
        df_posts = df["ShareCommentary"].dropna().reset_index(drop=True)
        cp = cache_path(user_name, "topic")
        topic_cache = load_cache(cp)

        uncached = [i for i in range(len(df_posts)) if str(i) not in topic_cache]
        if uncached:
            st.info(f"Classifying {len(uncached)} uncached posts with **{active_model}**…")
            print(f"[{active_model}] Starting topic classification for {len(uncached)} posts...")
            bar = st.progress(0, text="Starting…")
            status = st.empty()
            start_time = time.time()
            for idx, i in tqdm(enumerate(uncached), total=len(uncached), desc="Classifying posts"):
                t0 = time.time()
                topic_cache[str(i)] = classify_topic(df_posts.iloc[i], active_model)
                elapsed = time.time() - t0
                status.text(f"Post {i}: {elapsed:.1f}s | Processing {idx + 1}/{len(uncached)}…")
                print(f"  Post {i}: {elapsed:.1f}s")
                if idx % 5 == 0 or idx == len(uncached) - 1:
                    save_cache(cp, topic_cache)
                pct = (idx + 1) / len(uncached)
                bar.progress(pct, text=f"Topic classification: {idx + 1}/{len(uncached)}")
            save_cache(cp, topic_cache)
            total_time = time.time() - start_time
            print(f"Done in {total_time:.1f}s")
            bar.empty()
            status.empty()

        topic_labels = [topic_cache.get(str(i), "Other") for i in range(len(df_posts))]
        topic_counts = pd.Series(topic_labels).value_counts().reset_index()
        topic_counts.columns = ["topic", "count"]

        fig = px.bar(
            topic_counts, x="topic", y="count",
            title=f"LinkedIn Posts by Topic (Total: {len(df_posts)})",
            color="topic",
            text="count",
        )
        fig.update_layout(
            xaxis_title="Topic",
            yaxis_title="Number of Posts",
            showlegend=False,
            xaxis_tickangle=-30,
            bargap=0.2,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Company mentions ──────────────────────────────────────────────────────
    if show_companies:
        st.subheader("Company Mentions")
        df_orgs = df["ShareCommentary"].dropna().reset_index(drop=True)
        cp = cache_path(user_name, "company")
        company_cache = load_cache(cp)

        uncached = [i for i in range(len(df_orgs)) if str(i) not in company_cache]
        if uncached:
            st.info(f"Extracting companies from {len(uncached)} uncached posts with **{active_model}**…")
            print(f"[{active_model}] Starting company extraction for {len(uncached)} posts...")
            bar = st.progress(0, text="Starting…")
            status = st.empty()
            start_time = time.time()
            for idx, i in tqdm(enumerate(uncached), total=len(uncached), desc="Extracting companies"):
                t0 = time.time()
                company_cache[str(i)] = extract_companies(df_orgs.iloc[i], active_model)
                elapsed = time.time() - t0
                status.text(f"Post {i}: {elapsed:.1f}s | Processing {idx + 1}/{len(uncached)}…")
                print(f"  Post {i}: {elapsed:.1f}s")
                if idx % 5 == 0 or idx == len(uncached) - 1:
                    save_cache(cp, company_cache)
                pct = (idx + 1) / len(uncached)
                bar.progress(pct, text=f"Company extraction: {idx + 1}/{len(uncached)}")
            save_cache(cp, company_cache)
            total_time = time.time() - start_time
            print(f"Done in {total_time:.1f}s")
            bar.empty()
            status.empty()

        org_counter: Counter = Counter()
        for orgs in company_cache.values():
            for org in orgs:
                org_name = org.strip()
                if len(org_name) > 1:
                    org_counter[org_name] += 1

        top_n = 30
        top_orgs = pd.DataFrame(org_counter.most_common(top_n), columns=["org", "count"])

        fig = px.bar(
            top_orgs.sort_values("count"), x="count", y="org",
            orientation="h",
            title=f"Top {top_n} Companies/Organisations Mentioned in Posts",
            text="count",
            color="count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            xaxis_title="Mentions",
            yaxis_title="",
            coloraxis_showscale=False,
            height=800,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
